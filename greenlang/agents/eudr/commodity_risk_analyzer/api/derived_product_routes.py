# -*- coding: utf-8 -*-
"""
Derived Product Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for EUDR Annex I derived product analysis including product
analysis, processing chain retrieval, risk assessment, Annex I mapping,
and origin tracing.

Endpoints:
    POST /derived-products/analyze          - Analyze derived product
    GET  /derived-products/{product_id}/chain - Get processing chain
    GET  /derived-products/{product_id}/risk  - Get product risk
    GET  /derived-products/mapping           - Get Annex I mapping
    POST /derived-products/trace             - Trace product origin

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Derived Product Analysis Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_derived_product_analyzer,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity_type,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    AnnexIMappingEntry,
    AnnexIMappingResponse,
    CommodityTypeEnum,
    DerivedProductAnalyzeRequest,
    DerivedProductResponse,
    DerivedProductTraceRequest,
    ProcessingChainResponse,
    ProvenanceInfo,
    RiskLevelEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Derived Products"])

# ---------------------------------------------------------------------------
# In-memory store (replaced by database in production)
# ---------------------------------------------------------------------------

_product_store: Dict[str, DerivedProductResponse] = {}

# Static Annex I mapping data (subset for API demonstration)
_ANNEX_I_MAPPINGS: List[AnnexIMappingEntry] = [
    AnnexIMappingEntry(hs_code="1801.00", description="Cocoa beans, whole or broken, raw or roasted", source_commodity=CommodityTypeEnum.COCOA, product_category="cocoa_beans"),
    AnnexIMappingEntry(hs_code="1806.31", description="Chocolate, filled, in blocks/slabs/bars", source_commodity=CommodityTypeEnum.COCOA, product_category="chocolate"),
    AnnexIMappingEntry(hs_code="1806.32", description="Chocolate, not filled, in blocks/slabs/bars", source_commodity=CommodityTypeEnum.COCOA, product_category="chocolate"),
    AnnexIMappingEntry(hs_code="0901.11", description="Coffee, not roasted, not decaffeinated", source_commodity=CommodityTypeEnum.COFFEE, product_category="coffee_raw"),
    AnnexIMappingEntry(hs_code="0901.21", description="Coffee, roasted, not decaffeinated", source_commodity=CommodityTypeEnum.COFFEE, product_category="coffee_roasted"),
    AnnexIMappingEntry(hs_code="1511.10", description="Crude palm oil", source_commodity=CommodityTypeEnum.OIL_PALM, product_category="palm_oil"),
    AnnexIMappingEntry(hs_code="1511.90", description="Refined palm oil", source_commodity=CommodityTypeEnum.OIL_PALM, product_category="palm_oil"),
    AnnexIMappingEntry(hs_code="4001.21", description="Natural rubber sheets", source_commodity=CommodityTypeEnum.RUBBER, product_category="natural_rubber"),
    AnnexIMappingEntry(hs_code="4011.10", description="New pneumatic rubber tyres for motor cars", source_commodity=CommodityTypeEnum.RUBBER, product_category="tires"),
    AnnexIMappingEntry(hs_code="1201.90", description="Soya beans", source_commodity=CommodityTypeEnum.SOYA, product_category="soya_beans"),
    AnnexIMappingEntry(hs_code="2304.00", description="Soya-bean oil-cake and other solid residues", source_commodity=CommodityTypeEnum.SOYA, product_category="soy_meal"),
    AnnexIMappingEntry(hs_code="4403.11", description="Wood in the rough, treated with paint/stain", source_commodity=CommodityTypeEnum.WOOD, product_category="wood_rough"),
    AnnexIMappingEntry(hs_code="4407.11", description="Coniferous wood, sawn or chipped", source_commodity=CommodityTypeEnum.WOOD, product_category="lumber"),
    AnnexIMappingEntry(hs_code="4412.31", description="Plywood with tropical wood outer ply", source_commodity=CommodityTypeEnum.WOOD, product_category="plywood"),
    AnnexIMappingEntry(hs_code="0201.10", description="Carcasses and half-carcasses of bovine animals, fresh or chilled", source_commodity=CommodityTypeEnum.CATTLE, product_category="beef"),
    AnnexIMappingEntry(hs_code="4104.11", description="Full grains bovine leather, unsplit", source_commodity=CommodityTypeEnum.CATTLE, product_category="leather"),
]


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /derived-products/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/derived-products/analyze",
    response_model=DerivedProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze a derived product",
    description=(
        "Analyze an EUDR Annex I derived product including processing chain "
        "risk assessment, transformation ratio computation, traceability "
        "scoring, and Annex I reference mapping."
    ),
    responses={
        201: {"description": "Derived product analyzed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def analyze_derived_product(
    request: Request,
    body: DerivedProductAnalyzeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:derived-products:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DerivedProductResponse:
    """Analyze a derived product for EUDR Annex I compliance.

    Args:
        body: Product analysis request with processing stages.
        user: Authenticated user with derived-products:create permission.

    Returns:
        DerivedProductResponse with risk multiplier and traceability score.
    """
    start = time.monotonic()
    try:
        # Compute risk multiplier based on processing chain length and complexity
        chain_length = len(body.processing_stages)
        risk_multiplier = Decimal("1.0") + Decimal(str(chain_length)) * Decimal("0.15")
        risk_multiplier = min(risk_multiplier, Decimal("5.0"))

        # Compute traceability score (decreases with more stages)
        base_traceability = Decimal("1.0")
        for stage in body.processing_stages:
            base_traceability *= (Decimal("1.0") - stage.risk_contribution * Decimal("0.5"))
        traceability_score = max(Decimal("0.0"), base_traceability)

        # Look up Annex I reference
        annex_ref = None
        for mapping in _ANNEX_I_MAPPINGS:
            if mapping.source_commodity == body.source_commodity:
                annex_ref = mapping.hs_code
                break

        # Overall risk score
        overall_risk = min(
            Decimal("100.0"),
            Decimal("40.0") * risk_multiplier,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            body.model_dump_json(), f"{body.product_id}:{overall_risk}"
        )

        response = DerivedProductResponse(
            product_id=body.product_id,
            source_commodity=body.source_commodity,
            product_name=body.product_name,
            processing_chain=body.processing_stages,
            risk_multiplier=risk_multiplier.quantize(Decimal("0.01")),
            traceability_score=traceability_score.quantize(Decimal("0.01")),
            annex_i_reference=annex_ref,
            overall_risk_score=overall_risk.quantize(Decimal("0.01")),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

        _product_store[body.product_id] = response

        logger.info(
            "Derived product analyzed: id=%s source=%s risk_multiplier=%s",
            body.product_id,
            body.source_commodity.value,
            risk_multiplier,
        )

        return response

    except Exception as exc:
        logger.error("Derived product analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Derived product analysis failed",
        )


# ---------------------------------------------------------------------------
# GET /derived-products/{product_id}/chain
# ---------------------------------------------------------------------------


@router.get(
    "/derived-products/{product_id}/chain",
    response_model=ProcessingChainResponse,
    summary="Get processing chain",
    description="Retrieve the complete processing chain for a derived product.",
    responses={
        200: {"description": "Processing chain data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Product not found"},
    },
)
async def get_processing_chain(
    product_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:derived-products:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProcessingChainResponse:
    """Get processing chain for a derived product.

    Args:
        product_id: Product identifier.
        user: Authenticated user with derived-products:read permission.

    Returns:
        ProcessingChainResponse with ordered stages and transformation ratio.
    """
    product = _product_store.get(product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived product {product_id} not found",
        )

    return ProcessingChainResponse(
        source_commodity=product.source_commodity,
        final_product=product.product_name or product_id,
        stages=product.processing_chain,
        total_risk=product.overall_risk_score,
        transformation_ratio=product.risk_multiplier,
        chain_length=len(product.processing_chain),
    )


# ---------------------------------------------------------------------------
# GET /derived-products/{product_id}/risk
# ---------------------------------------------------------------------------


@router.get(
    "/derived-products/{product_id}/risk",
    response_model=DerivedProductResponse,
    summary="Get derived product risk",
    description="Retrieve risk assessment results for a specific derived product.",
    responses={
        200: {"description": "Product risk data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Product not found"},
    },
)
async def get_product_risk(
    product_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:derived-products:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DerivedProductResponse:
    """Get risk assessment for a derived product.

    Args:
        product_id: Product identifier.
        user: Authenticated user with derived-products:read permission.

    Returns:
        DerivedProductResponse with risk scores and provenance.
    """
    product = _product_store.get(product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived product {product_id} not found",
        )
    return product


# ---------------------------------------------------------------------------
# GET /derived-products/mapping
# ---------------------------------------------------------------------------


@router.get(
    "/derived-products/mapping",
    response_model=AnnexIMappingResponse,
    summary="Get Annex I mapping",
    description=(
        "Retrieve the EUDR Annex I product-to-commodity mapping table, "
        "optionally filtered by commodity type. Maps HS tariff codes to "
        "EUDR-regulated source commodities."
    ),
    responses={
        200: {"description": "Annex I mappings"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_annex_i_mapping(
    request: Request,
    commodity_type: Optional[str] = Depends(validate_commodity_type),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:derived-products:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AnnexIMappingResponse:
    """Get EUDR Annex I product-to-commodity mappings.

    Args:
        commodity_type: Optional commodity type filter.
        user: Authenticated user with derived-products:read permission.

    Returns:
        AnnexIMappingResponse with filtered Annex I entries.
    """
    mappings = _ANNEX_I_MAPPINGS
    filter_enum = None
    if commodity_type:
        filter_enum = CommodityTypeEnum(commodity_type)
        mappings = [m for m in mappings if m.source_commodity == filter_enum]

    return AnnexIMappingResponse(
        mappings=mappings,
        total_entries=len(mappings),
        commodity_filter=filter_enum,
    )


# ---------------------------------------------------------------------------
# POST /derived-products/trace
# ---------------------------------------------------------------------------


@router.post(
    "/derived-products/trace",
    response_model=DerivedProductResponse,
    summary="Trace product origin",
    description=(
        "Trace a derived product back to its raw commodity origin through "
        "the complete processing chain for EUDR traceability compliance."
    ),
    responses={
        200: {"description": "Product trace results"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Product not found"},
    },
)
async def trace_product_origin(
    request: Request,
    body: DerivedProductTraceRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:derived-products:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DerivedProductResponse:
    """Trace a derived product back to its raw commodity origin.

    Args:
        body: Trace request with product identifier.
        user: Authenticated user with derived-products:read permission.

    Returns:
        DerivedProductResponse with full traceability chain.

    Raises:
        HTTPException: 404 if product not found.
    """
    product = _product_store.get(body.product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Derived product {body.product_id} not found",
        )

    logger.info(
        "Product traced: id=%s source=%s chain_length=%d",
        body.product_id,
        product.source_commodity.value,
        len(product.processing_chain),
    )

    return product
