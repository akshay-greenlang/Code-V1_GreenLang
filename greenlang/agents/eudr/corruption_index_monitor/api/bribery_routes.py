# -*- coding: utf-8 -*-
"""
Bribery Risk Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for sector-specific bribery risk assessment covering 6 EUDR-relevant
sectors: forestry, customs, agriculture, mining, extraction, and judiciary.
Provides composite bribery risk scoring, per-sector breakdowns, high-risk
country identification, and cross-country sector analysis.

Endpoints:
    POST /bribery/assess                  - Assess bribery risk for a country
    GET  /bribery/{country_code}/risk     - Get country bribery risk profile
    GET  /bribery/{country_code}/sectors  - Get sector-specific risks
    GET  /bribery/high-risk-countries     - List high-risk bribery countries
    GET  /bribery/sector-analysis         - Cross-country sector analysis

Sectors: forestry (0.25), customs (0.20), agriculture (0.20), mining (0.15),
         extraction (0.10), judiciary (0.10)
Data Source: TRACE Bribery Risk Matrix

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Bribery Risk Engine
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
    PaginationParams,
    get_bribery_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    BriberyAssessmentRequest,
    BriberyAssessmentResponse,
    BriberyProfileResponse,
    BriberySectorEnum,
    BriberySectorScore,
    DataSourceEnum,
    ErrorResponse as SchemaErrorResponse,
    HighRiskCountriesResponse,
    HighRiskCountryEntry,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
    SectorExposureCountryEntry,
    SectorExposureResponse,
    SectorRiskEntry,
    SectorRiskResponse,
    TrendDirectionEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bribery", tags=["Bribery Risk Assessment"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_bribery_risk(score: Decimal) -> RiskLevelEnum:
    """Classify bribery risk score into a risk level.

    Args:
        score: Bribery risk score on 0-100 scale (higher=more risk).

    Returns:
        Risk level classification.
    """
    if score >= Decimal("75"):
        return RiskLevelEnum.CRITICAL
    elif score >= Decimal("50"):
        return RiskLevelEnum.HIGH
    elif score >= Decimal("25"):
        return RiskLevelEnum.MODERATE
    return RiskLevelEnum.LOW


# ---------------------------------------------------------------------------
# POST /bribery/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=BriberyAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess bribery risk for a country",
    description=(
        "Perform a comprehensive bribery risk assessment for a country across "
        "6 EUDR-relevant sectors: forestry, customs, agriculture, mining, "
        "extraction, and judiciary. Returns weighted composite score and "
        "per-sector breakdowns with contributing factors and mitigation measures."
    ),
    responses={
        200: {"description": "Bribery assessment completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_bribery_risk(
    request: Request,
    body: BriberyAssessmentRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:bribery:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BriberyAssessmentResponse:
    """Assess bribery risk for a country across EUDR-relevant sectors.

    Args:
        body: Bribery assessment request with country and sector selection.
        user: Authenticated user with bribery:create permission.

    Returns:
        BriberyAssessmentResponse with composite and per-sector scores.
    """
    start = time.monotonic()

    try:
        engine = get_bribery_engine()
        result = engine.assess_risk(
            country_code=body.country_code,
            sectors=[s.value for s in body.sectors] if body.sectors else None,
            include_mitigation=body.include_mitigation,
            commodity_type=body.commodity_type,
        )

        sector_scores = []
        for ss in result.get("sector_scores", []):
            score_val = Decimal(str(ss.get("risk_score", 0)))
            sector_scores.append(
                BriberySectorScore(
                    sector=BriberySectorEnum(ss.get("sector", "forestry")),
                    risk_score=score_val,
                    risk_level=_classify_bribery_risk(score_val),
                    weight=Decimal(str(ss.get("weight", 0.1))),
                    contributing_factors=ss.get("contributing_factors", []),
                    mitigation_measures=ss.get("mitigation_measures", []),
                )
            )

        composite = Decimal(str(result.get("composite_bribery_risk", 0)))
        highest_sector = BriberySectorEnum(result.get("highest_risk_sector", "forestry"))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"bribery:{body.country_code}", str(composite)
        )

        logger.info(
            "Bribery assessment completed: country=%s composite=%s operator=%s",
            body.country_code,
            composite,
            user.operator_id or user.user_id,
        )

        return BriberyAssessmentResponse(
            country_code=body.country_code,
            country_name=result.get("country_name", ""),
            composite_bribery_risk=composite,
            risk_level=_classify_bribery_risk(composite),
            sector_scores=sector_scores,
            highest_risk_sector=highest_sector,
            cpi_correlation=Decimal(str(result.get("cpi_correlation", 0))) if result.get("cpi_correlation") else None,
            data_source=DataSourceEnum.TRACE_MATRIX,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["TRACE Bribery Risk Matrix"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bribery assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bribery risk assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /bribery/{country_code}/risk
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/risk",
    response_model=BriberyProfileResponse,
    summary="Get country bribery risk profile",
    description=(
        "Retrieve the comprehensive bribery risk profile for a country "
        "including sector breakdown, historical trend, and peer comparison."
    ),
    responses={
        200: {"description": "Bribery profile retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_bribery_risk_profile(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:bribery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BriberyProfileResponse:
    """Get the bribery risk profile for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with bribery:read permission.

    Returns:
        BriberyProfileResponse with full risk profile.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_bribery_engine()
        result = engine.get_profile(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bribery risk profile not found for {normalized_code}",
            )

        sector_breakdown = []
        for ss in result.get("sector_breakdown", []):
            score_val = Decimal(str(ss.get("risk_score", 0)))
            sector_breakdown.append(
                BriberySectorScore(
                    sector=BriberySectorEnum(ss.get("sector", "forestry")),
                    risk_score=score_val,
                    risk_level=_classify_bribery_risk(score_val),
                    weight=Decimal(str(ss.get("weight", 0.1))),
                    contributing_factors=ss.get("contributing_factors", []),
                    mitigation_measures=ss.get("mitigation_measures", []),
                )
            )

        overall_risk = Decimal(str(result.get("overall_bribery_risk", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"bribery_profile:{normalized_code}", str(overall_risk)
        )

        logger.info(
            "Bribery profile retrieved: country=%s risk=%s operator=%s",
            normalized_code,
            overall_risk,
            user.operator_id or user.user_id,
        )

        return BriberyProfileResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            overall_bribery_risk=overall_risk,
            risk_level=_classify_bribery_risk(overall_risk),
            sector_breakdown=sector_breakdown,
            historical_trend=TrendDirectionEnum(result.get("historical_trend", "stable")),
            peer_comparison_percentile=Decimal(str(result.get("peer_percentile", 50))) if result.get("peer_percentile") else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["TRACE Bribery Risk Matrix"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bribery profile retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bribery risk profile retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /bribery/{country_code}/sectors
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/sectors",
    response_model=SectorRiskResponse,
    summary="Get sector-specific bribery risks for a country",
    description=(
        "Retrieve bribery risk scores broken down by sector for a specific "
        "country. Highlights forestry sector risk as EUDR-critical."
    ),
    responses={
        200: {"description": "Sector risks retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_sector_risks(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:bribery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SectorRiskResponse:
    """Get sector-specific bribery risk breakdown for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with bribery:read permission.

    Returns:
        SectorRiskResponse with per-sector risk scores.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_bribery_engine()
        result = engine.get_sector_risks(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sector risks not found for {normalized_code}",
            )

        sectors = []
        forestry_detail = None
        highest_risk = BriberySectorEnum.FORESTRY
        highest_score = Decimal("0")

        for sr in result.get("sectors", []):
            score_val = Decimal(str(sr.get("risk_score", 0)))
            sector_enum = BriberySectorEnum(sr.get("sector", "forestry"))
            entry = SectorRiskEntry(
                sector=sector_enum,
                risk_score=score_val,
                risk_level=_classify_bribery_risk(score_val),
                eudr_relevance=Decimal(str(sr.get("eudr_relevance", 0.5))),
            )
            sectors.append(entry)

            if score_val > highest_score:
                highest_score = score_val
                highest_risk = sector_enum

            if sector_enum == BriberySectorEnum.FORESTRY:
                forestry_detail = BriberySectorScore(
                    sector=BriberySectorEnum.FORESTRY,
                    risk_score=score_val,
                    risk_level=_classify_bribery_risk(score_val),
                    weight=Decimal(str(sr.get("weight", 0.25))),
                    contributing_factors=sr.get("contributing_factors", []),
                    mitigation_measures=sr.get("mitigation_measures", []),
                )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"sector_risks:{normalized_code}", str(highest_score)
        )

        logger.info(
            "Sector risks retrieved: country=%s sectors=%d operator=%s",
            normalized_code,
            len(sectors),
            user.operator_id or user.user_id,
        )

        return SectorRiskResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            sectors=sectors,
            highest_risk_sector=highest_risk,
            forestry_risk_detail=forestry_detail,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["TRACE Bribery Risk Matrix"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Sector risks retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sector risks retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /bribery/high-risk-countries
# ---------------------------------------------------------------------------


@router.get(
    "/high-risk-countries",
    response_model=HighRiskCountriesResponse,
    summary="List countries with high bribery risk",
    description=(
        "Retrieve a paginated list of countries exceeding the bribery risk "
        "threshold, sorted by risk score descending. Includes EUDR commodity "
        "exposure per country."
    ),
    responses={
        200: {"description": "High-risk countries retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_high_risk_countries(
    request: Request,
    threshold: Decimal = Query(
        default=Decimal("60"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Minimum bribery risk score threshold",
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:bribery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> HighRiskCountriesResponse:
    """Get list of countries exceeding the bribery risk threshold.

    Args:
        threshold: Minimum bribery risk score for inclusion.
        pagination: Pagination parameters.
        user: Authenticated user with bribery:read permission.

    Returns:
        HighRiskCountriesResponse with filtered high-risk countries.
    """
    start = time.monotonic()

    try:
        engine = get_bribery_engine()
        result = engine.get_high_risk_countries(
            threshold=float(threshold),
            limit=pagination.limit,
            offset=pagination.offset,
        )

        countries = []
        for entry in result.get("countries", []):
            risk_score = Decimal(str(entry.get("composite_bribery_risk", 0)))
            countries.append(
                HighRiskCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    composite_bribery_risk=risk_score,
                    risk_level=_classify_bribery_risk(risk_score),
                    highest_risk_sector=BriberySectorEnum(entry.get("highest_risk_sector", "forestry")),
                    eudr_commodity_exposure=entry.get("eudr_commodity_exposure", []),
                )
            )

        total = result.get("total", len(countries))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"high_risk:{threshold}", str(total)
        )

        logger.info(
            "High-risk countries retrieved: threshold=%s total=%d operator=%s",
            threshold,
            total,
            user.operator_id or user.user_id,
        )

        return HighRiskCountriesResponse(
            threshold=threshold,
            countries=countries,
            total_high_risk=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["TRACE Bribery Risk Matrix"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("High-risk countries retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="High-risk countries retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /bribery/sector-analysis
# ---------------------------------------------------------------------------


@router.get(
    "/sector-analysis",
    response_model=SectorExposureResponse,
    summary="Cross-country sector bribery risk analysis",
    description=(
        "Analyze bribery risk across countries for a specific sector. "
        "Returns per-country sector risk scores with global averages."
    ),
    responses={
        200: {"description": "Sector analysis completed"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_sector_analysis(
    request: Request,
    sector: BriberySectorEnum = Query(
        ...,
        description="Sector to analyze across countries",
    ),
    year: Optional[int] = Query(None, ge=2000, le=2030, description="Analysis year"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:bribery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SectorExposureResponse:
    """Analyze bribery risk across countries for a specific sector.

    Args:
        sector: Sector to analyze.
        year: Analysis year (default: latest).
        pagination: Pagination parameters.
        user: Authenticated user with bribery:read permission.

    Returns:
        SectorExposureResponse with cross-country sector risk data.
    """
    start = time.monotonic()

    try:
        engine = get_bribery_engine()
        result = engine.get_sector_analysis(
            sector=sector.value,
            year=year,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        countries = []
        for entry in result.get("countries", []):
            score_val = Decimal(str(entry.get("sector_risk_score", 0)))
            countries.append(
                SectorExposureCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    sector_risk_score=score_val,
                    risk_level=_classify_bribery_risk(score_val),
                )
            )

        total = result.get("total", len(countries))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"sector_analysis:{sector.value}:{result_year}", str(total)
        )

        logger.info(
            "Sector analysis completed: sector=%s year=%d countries=%d operator=%s",
            sector.value,
            result_year,
            total,
            user.operator_id or user.user_id,
        )

        return SectorExposureResponse(
            sector=sector,
            year=result_year,
            countries=countries,
            sector_global_average=Decimal(str(result.get("global_average", 0))),
            high_risk_count=result.get("high_risk_count", 0),
            total_countries=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["TRACE Bribery Risk Matrix"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Sector analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sector analysis failed",
        )
