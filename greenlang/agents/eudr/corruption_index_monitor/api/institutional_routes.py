# -*- coding: utf-8 -*-
"""
Institutional Quality Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for institutional quality assessment covering judicial independence,
regulatory enforcement, forest governance, and law enforcement capacity.
Provides composite quality scoring, detailed governance profiles, strength
assessments, forest-specific governance metrics, and country comparison.

Endpoints:
    GET  /institutional/{country_code}/quality          - Quality assessment
    GET  /institutional/{country_code}/governance        - Governance profile
    POST /institutional/assess                          - Strength assessment
    GET  /institutional/{country_code}/forest-governance - Forest governance
    POST /institutional/compare                         - Compare countries

Dimensions: judicial_independence (0.30), regulatory_enforcement (0.25),
            forest_governance (0.25), law_enforcement_capacity (0.20)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Institutional Quality Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.corruption_index_monitor.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_institutional_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    ErrorResponse as SchemaErrorResponse,
    ForestGovernanceResponse,
    GovernanceProfileResponse,
    GovernanceRatingEnum,
    InstitutionalComparisonCountryEntry,
    InstitutionalComparisonRequest,
    InstitutionalComparisonResponse,
    InstitutionalDimensionScore,
    InstitutionalQualityResponse,
    MetadataSchema,
    ProvenanceInfo,
    RiskLevelEnum,
    StrengthAssessmentRequest,
    StrengthAssessmentResponse,
    TrendDirectionEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/institutional", tags=["Institutional Quality"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_iq_risk(score: Decimal) -> RiskLevelEnum:
    """Classify institutional quality score into a risk level."""
    if score >= Decimal("75"):
        return RiskLevelEnum.LOW
    elif score >= Decimal("50"):
        return RiskLevelEnum.MODERATE
    elif score >= Decimal("25"):
        return RiskLevelEnum.HIGH
    return RiskLevelEnum.CRITICAL


def _rate_governance(score: Decimal) -> GovernanceRatingEnum:
    """Rate governance quality based on institutional score."""
    if score >= Decimal("80"):
        return GovernanceRatingEnum.EXCELLENT
    elif score >= Decimal("60"):
        return GovernanceRatingEnum.GOOD
    elif score >= Decimal("40"):
        return GovernanceRatingEnum.ADEQUATE
    elif score >= Decimal("20"):
        return GovernanceRatingEnum.POOR
    return GovernanceRatingEnum.CRITICAL


# ---------------------------------------------------------------------------
# GET /institutional/{country_code}/quality
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/quality",
    response_model=InstitutionalQualityResponse,
    summary="Get institutional quality assessment for a country",
    description=(
        "Retrieve composite institutional quality score for a country "
        "covering judicial independence, regulatory enforcement, forest "
        "governance, and law enforcement capacity. Identifies strongest "
        "and weakest institutional dimensions."
    ),
    responses={
        200: {"description": "Quality assessment retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_institutional_quality(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:institutional:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> InstitutionalQualityResponse:
    """Get institutional quality assessment for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with institutional:read permission.

    Returns:
        InstitutionalQualityResponse with composite and per-dimension scores.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_institutional_engine()
        result = engine.assess_quality(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Institutional quality data not found for {normalized_code}",
            )

        dimensions = []
        for dim_data in result.get("dimensions", []):
            score_val = Decimal(str(dim_data.get("score", 0)))
            dimensions.append(
                InstitutionalDimensionScore(
                    dimension=dim_data.get("dimension", ""),
                    score=score_val,
                    weight=Decimal(str(dim_data.get("weight", 0.25))),
                    rating=_rate_governance(score_val),
                    indicators=dim_data.get("indicators", []),
                )
            )

        composite = Decimal(str(result.get("composite_score", 0)))
        strongest = result.get("strongest_dimension", "")
        weakest = result.get("weakest_dimension", "")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"iq:{normalized_code}", str(composite)
        )

        logger.info(
            "Institutional quality retrieved: country=%s composite=%s operator=%s",
            normalized_code,
            composite,
            user.operator_id or user.user_id,
        )

        return InstitutionalQualityResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            composite_score=composite,
            governance_rating=_rate_governance(composite),
            dimensions=dimensions,
            strongest_dimension=strongest,
            weakest_dimension=weakest,
            risk_level=_classify_iq_risk(composite),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI", "Internal Assessment"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Institutional quality retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institutional quality assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /institutional/{country_code}/governance
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/governance",
    response_model=GovernanceProfileResponse,
    summary="Get detailed governance profile for a country",
    description=(
        "Retrieve a comprehensive governance profile for a country including "
        "institutional quality scores, CPI/WGI context, detailed dimension "
        "breakdowns, historical trend, and peer country benchmarks."
    ),
    responses={
        200: {"description": "Governance profile retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_governance_profile(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:institutional:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> GovernanceProfileResponse:
    """Get detailed governance profile for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with institutional:read permission.

    Returns:
        GovernanceProfileResponse with comprehensive governance data.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_institutional_engine()
        result = engine.get_governance_profile(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Governance profile not found for {normalized_code}",
            )

        dimensions = []
        for dim_data in result.get("dimensions", []):
            score_val = Decimal(str(dim_data.get("score", 0)))
            dimensions.append(
                InstitutionalDimensionScore(
                    dimension=dim_data.get("dimension", ""),
                    score=score_val,
                    weight=Decimal(str(dim_data.get("weight", 0.25))),
                    rating=_rate_governance(score_val),
                    indicators=dim_data.get("indicators", []),
                )
            )

        composite = Decimal(str(result.get("institutional_quality_score", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"governance:{normalized_code}", str(composite)
        )

        logger.info(
            "Governance profile retrieved: country=%s score=%s operator=%s",
            normalized_code,
            composite,
            user.operator_id or user.user_id,
        )

        return GovernanceProfileResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            governance_rating=_rate_governance(composite),
            institutional_quality_score=composite,
            cpi_score=Decimal(str(result.get("cpi_score", 0))) if result.get("cpi_score") else None,
            wgi_composite=Decimal(str(result.get("wgi_composite", 0))) if result.get("wgi_composite") else None,
            dimensions=dimensions,
            judicial_independence_detail=result.get("judicial_independence_detail"),
            regulatory_enforcement_detail=result.get("regulatory_enforcement_detail"),
            forest_governance_detail=result.get("forest_governance_detail"),
            law_enforcement_detail=result.get("law_enforcement_detail"),
            historical_trend=TrendDirectionEnum(result.get("historical_trend", "stable")),
            peer_countries=result.get("peer_countries", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI", "Transparency International", "Internal Assessment"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Governance profile retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Governance profile retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /institutional/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=StrengthAssessmentResponse,
    summary="Perform institutional strength assessment",
    description=(
        "Run a comprehensive institutional strength assessment including "
        "per-dimension scoring, strengths/weaknesses identification, "
        "improvement recommendations, and optional regional benchmarking."
    ),
    responses={
        200: {"description": "Strength assessment completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_institutional_strength(
    request: Request,
    body: StrengthAssessmentRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:institutional:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> StrengthAssessmentResponse:
    """Perform institutional strength assessment for a country.

    Args:
        body: Strength assessment request with country and options.
        user: Authenticated user with institutional:create permission.

    Returns:
        StrengthAssessmentResponse with scores and recommendations.
    """
    start = time.monotonic()

    try:
        engine = get_institutional_engine()
        result = engine.assess_strength(
            country_code=body.country_code,
            dimensions=body.dimensions,
            include_recommendations=body.include_recommendations,
            benchmark_region=body.benchmark_region.value if body.benchmark_region else None,
        )

        dimensions = []
        for dim_data in result.get("dimensions", []):
            score_val = Decimal(str(dim_data.get("score", 0)))
            dimensions.append(
                InstitutionalDimensionScore(
                    dimension=dim_data.get("dimension", ""),
                    score=score_val,
                    weight=Decimal(str(dim_data.get("weight", 0.25))),
                    rating=_rate_governance(score_val),
                    indicators=dim_data.get("indicators", []),
                )
            )

        composite = Decimal(str(result.get("composite_strength", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"strength:{body.country_code}", str(composite)
        )

        logger.info(
            "Strength assessment completed: country=%s composite=%s operator=%s",
            body.country_code,
            composite,
            user.operator_id or user.user_id,
        )

        return StrengthAssessmentResponse(
            country_code=body.country_code,
            country_name=result.get("country_name", ""),
            composite_strength=composite,
            governance_rating=_rate_governance(composite),
            dimensions=dimensions,
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            recommendations=result.get("recommendations", []),
            benchmark_percentile=Decimal(str(result.get("benchmark_percentile", 50))) if result.get("benchmark_percentile") else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI", "Internal Assessment"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Strength assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institutional strength assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /institutional/{country_code}/forest-governance
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/forest-governance",
    response_model=ForestGovernanceResponse,
    summary="Get forest governance assessment for a country",
    description=(
        "Retrieve forest-specific governance assessment including legal "
        "framework strength, enforcement capacity, monitoring capability, "
        "transparency, corruption vulnerability, and EUDR readiness."
    ),
    responses={
        200: {"description": "Forest governance assessment retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_forest_governance(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:institutional:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ForestGovernanceResponse:
    """Get forest governance assessment for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with institutional:read permission.

    Returns:
        ForestGovernanceResponse with forest-specific governance metrics.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_institutional_engine()
        result = engine.get_forest_governance(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Forest governance data not found for {normalized_code}",
            )

        fg_score = Decimal(str(result.get("forest_governance_score", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"forest_gov:{normalized_code}", str(fg_score)
        )

        logger.info(
            "Forest governance retrieved: country=%s score=%s operator=%s",
            normalized_code,
            fg_score,
            user.operator_id or user.user_id,
        )

        return ForestGovernanceResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            forest_governance_score=fg_score,
            governance_rating=_rate_governance(fg_score),
            legal_framework_strength=Decimal(str(result.get("legal_framework_strength", 0))),
            enforcement_capacity=Decimal(str(result.get("enforcement_capacity", 0))),
            monitoring_capability=Decimal(str(result.get("monitoring_capability", 0))),
            transparency_score=Decimal(str(result.get("transparency_score", 0))),
            corruption_vulnerability=Decimal(str(result.get("corruption_vulnerability", 0))),
            eudr_readiness=Decimal(str(result.get("eudr_readiness", 0))),
            key_risks=result.get("key_risks", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI", "Global Forest Watch", "Internal Assessment"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Forest governance retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forest governance assessment failed",
        )


# ---------------------------------------------------------------------------
# POST /institutional/compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=InstitutionalComparisonResponse,
    summary="Compare institutional quality across countries",
    description=(
        "Compare institutional quality across 2-20 countries. Returns "
        "per-country composite scores, dimension breakdowns, pairwise "
        "differential matrix, and dimension-level leaders."
    ),
    responses={
        200: {"description": "Institutional comparison completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_institutional(
    request: Request,
    body: InstitutionalComparisonRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:institutional:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> InstitutionalComparisonResponse:
    """Compare institutional quality across countries.

    Args:
        body: Comparison request with country codes and optional dimension filter.
        user: Authenticated user with institutional:read permission.

    Returns:
        InstitutionalComparisonResponse with comparison results.
    """
    start = time.monotonic()

    try:
        engine = get_institutional_engine()
        result = engine.compare_countries(
            country_codes=body.country_codes,
            dimensions=body.dimensions,
        )

        countries = []
        for entry in result.get("countries", []):
            dims = []
            for dim_data in entry.get("dimensions", []):
                score_val = Decimal(str(dim_data.get("score", 0)))
                dims.append(
                    InstitutionalDimensionScore(
                        dimension=dim_data.get("dimension", ""),
                        score=score_val,
                        weight=Decimal(str(dim_data.get("weight", 0.25))),
                        rating=_rate_governance(score_val),
                    )
                )
            composite = Decimal(str(entry.get("composite_score", 0)))
            countries.append(
                InstitutionalComparisonCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    composite_score=composite,
                    governance_rating=_rate_governance(composite),
                    dimensions=dims,
                )
            )

        matrix: Dict[str, Dict[str, Decimal]] = {}
        for c1 in countries:
            matrix[c1.country_code] = {}
            for c2 in countries:
                matrix[c1.country_code][c2.country_code] = abs(
                    c1.composite_score - c2.composite_score
                ).quantize(Decimal("0.01"))

        best = max(countries, key=lambda c: c.composite_score).country_code if countries else None
        worst = min(countries, key=lambda c: c.composite_score).country_code if countries else None

        dimension_leaders = result.get("dimension_leaders", {})

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"iq_compare:{','.join(body.country_codes)}",
            str(len(countries)),
        )

        logger.info(
            "Institutional comparison completed: countries=%d operator=%s",
            len(countries),
            user.operator_id or user.user_id,
        )

        return InstitutionalComparisonResponse(
            countries=countries,
            differential_matrix=matrix,
            best_performer=best,
            worst_performer=worst,
            dimension_leaders=dimension_leaders,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI", "Internal Assessment"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Institutional comparison failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institutional comparison failed",
        )
