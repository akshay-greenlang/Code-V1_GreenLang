# -*- coding: utf-8 -*-
"""
Compliance Impact Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for EUDR compliance impact assessment mapping corruption indices
to country classifications per EUDR Article 29 and determining due diligence
requirements (simplified, standard, enhanced).

Endpoints:
    POST /compliance/assess-impact          - Full compliance impact assessment
    GET  /compliance/{country_code}/impact  - Country impact profile
    GET  /compliance/dd-recommendations     - Due diligence recommendations
    GET  /compliance/country-classifications - EUDR country classifications

Classification thresholds:
    Low risk:     CPI >= 60 AND WGI >= 0.5  -> Simplified DD
    Standard risk: Between low and high      -> Standard DD
    High risk:    CPI <= 30 OR WGI <= -0.5  -> Enhanced DD

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Compliance Impact Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.corruption_index_monitor.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_compliance_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    ComplianceImpactRequest,
    ComplianceImpactResponse,
    ComplianceLevelEnum,
    CountryClassificationEntry,
    CountryClassificationEnum,
    CountryClassificationResponse,
    CountryImpactResponse,
    DDRecommendationEntry,
    DDRecommendationsResponse,
    DueDiligenceCostEstimate,
    ErrorResponse as SchemaErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RegionEnum,
    RiskLevelEnum,
    TrendDirectionEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance Impact"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_risk(classification: CountryClassificationEnum) -> RiskLevelEnum:
    """Map country classification to risk level."""
    mapping = {
        CountryClassificationEnum.LOW_RISK: RiskLevelEnum.LOW,
        CountryClassificationEnum.STANDARD_RISK: RiskLevelEnum.MODERATE,
        CountryClassificationEnum.HIGH_RISK: RiskLevelEnum.CRITICAL,
    }
    return mapping.get(classification, RiskLevelEnum.MODERATE)


def _map_dd_level(classification: CountryClassificationEnum) -> ComplianceLevelEnum:
    """Map country classification to due diligence level."""
    mapping = {
        CountryClassificationEnum.LOW_RISK: ComplianceLevelEnum.SIMPLIFIED,
        CountryClassificationEnum.STANDARD_RISK: ComplianceLevelEnum.STANDARD,
        CountryClassificationEnum.HIGH_RISK: ComplianceLevelEnum.ENHANCED,
    }
    return mapping.get(classification, ComplianceLevelEnum.STANDARD)


# ---------------------------------------------------------------------------
# POST /compliance/assess-impact
# ---------------------------------------------------------------------------


@router.post(
    "/assess-impact",
    response_model=ComplianceImpactResponse,
    summary="Assess EUDR compliance impact for a country",
    description=(
        "Perform a comprehensive EUDR compliance impact assessment for a "
        "country based on corruption indices. Determines Article 29 country "
        "classification, required due diligence level, cost estimates, and "
        "compliance recommendations."
    ),
    responses={
        200: {"description": "Compliance impact assessed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_compliance_impact(
    request: Request,
    body: ComplianceImpactRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:compliance:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ComplianceImpactResponse:
    """Assess EUDR compliance impact for a country.

    Args:
        body: Compliance impact assessment request.
        user: Authenticated user with compliance:create permission.

    Returns:
        ComplianceImpactResponse with classification and recommendations.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.assess_impact(
            country_code=body.country_code,
            commodity_types=body.commodity_types,
            include_cost_estimates=body.include_cost_estimates,
            include_recommendations=body.include_recommendations,
        )

        classification = CountryClassificationEnum(
            result.get("eudr_classification", "standard_risk")
        )
        dd_level = _map_dd_level(classification)

        cost_estimate = None
        cost_data = result.get("cost_estimates")
        if cost_data and body.include_cost_estimates:
            cost_estimate = DueDiligenceCostEstimate(
                dd_level=dd_level,
                estimated_cost_eur=Decimal(str(cost_data.get("estimated_cost_eur", 0))),
                audit_frequency_months=cost_data.get("audit_frequency_months", 12),
                estimated_duration_days=cost_data.get("estimated_duration_days", 30),
                required_resources=cost_data.get("required_resources", []),
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_impact:{body.country_code}",
            str(classification.value),
        )

        logger.info(
            "Compliance impact assessed: country=%s classification=%s dd=%s operator=%s",
            body.country_code,
            classification.value,
            dd_level.value,
            user.operator_id or user.user_id,
        )

        return ComplianceImpactResponse(
            country_code=body.country_code,
            country_name=result.get("country_name", ""),
            eudr_classification=classification,
            required_dd_level=dd_level,
            cpi_score=Decimal(str(result.get("cpi_score", 0))) if result.get("cpi_score") else None,
            wgi_composite=Decimal(str(result.get("wgi_composite", 0))) if result.get("wgi_composite") else None,
            risk_factors=result.get("risk_factors", []),
            mitigating_factors=result.get("mitigating_factors", []),
            cost_estimates=cost_estimate,
            recommendations=result.get("recommendations", []),
            regulatory_articles=result.get("regulatory_articles", ["Art. 10", "Art. 11", "Art. 29"]),
            classification_rationale=result.get("classification_rationale", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "Transparency International CPI",
                    "World Bank WGI",
                    "EUDR Article 29 Criteria",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance impact assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance impact assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/{country_code}/impact
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/impact",
    response_model=CountryImpactResponse,
    summary="Get country compliance impact profile",
    description=(
        "Retrieve the current EUDR compliance impact profile for a country "
        "including classification, due diligence level, corruption indices, "
        "risk trajectory, and active alert count."
    ),
    responses={
        200: {"description": "Country impact profile retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_country_impact(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CountryImpactResponse:
    """Get the compliance impact profile for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with compliance:read permission.

    Returns:
        CountryImpactResponse with compliance impact profile.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_compliance_engine()
        result = engine.get_country_impact(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Compliance impact data not found for {normalized_code}",
            )

        classification = CountryClassificationEnum(
            result.get("eudr_classification", "standard_risk")
        )
        dd_level = _map_dd_level(classification)

        corruption_indices = {}
        for k, v in result.get("corruption_indices", {}).items():
            corruption_indices[k] = Decimal(str(v))

        prev_class = result.get("previous_classification")
        prev_classification = CountryClassificationEnum(prev_class) if prev_class else None

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"country_impact:{normalized_code}", str(classification.value)
        )

        logger.info(
            "Country impact retrieved: country=%s classification=%s operator=%s",
            normalized_code,
            classification.value,
            user.operator_id or user.user_id,
        )

        return CountryImpactResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            eudr_classification=classification,
            previous_classification=prev_classification,
            classification_changed=result.get("classification_changed", False),
            change_date=result.get("change_date"),
            required_dd_level=dd_level,
            corruption_indices=corruption_indices,
            risk_trajectory=TrendDirectionEnum(result.get("risk_trajectory", "stable")),
            next_review_date=result.get("next_review_date"),
            active_alerts=result.get("active_alerts", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "Transparency International CPI",
                    "World Bank WGI",
                    "EUDR Article 29 Criteria",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Country impact retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Country impact retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/dd-recommendations
# ---------------------------------------------------------------------------


@router.get(
    "/dd-recommendations",
    response_model=DDRecommendationsResponse,
    summary="Get due diligence recommendations",
    description=(
        "Retrieve prioritized due diligence recommendations based on the "
        "required DD level. Optionally filter by country for country-specific "
        "recommendations."
    ),
    responses={
        200: {"description": "DD recommendations retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_dd_recommendations(
    request: Request,
    dd_level: ComplianceLevelEnum = Query(
        ...,
        description="Due diligence level for recommendations",
    ),
    country_code: Optional[str] = Query(
        None,
        description="Optional country code for country-specific recommendations",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DDRecommendationsResponse:
    """Get due diligence recommendations for a specific DD level.

    Args:
        dd_level: Due diligence level.
        country_code: Optional country code for context.
        user: Authenticated user with compliance:read permission.

    Returns:
        DDRecommendationsResponse with prioritized recommendations.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.get_dd_recommendations(
            dd_level=dd_level.value,
            country_code=country_code.upper() if country_code else None,
        )

        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append(
                DDRecommendationEntry(
                    recommendation_id=rec.get("recommendation_id", ""),
                    category=rec.get("category", "documentation"),
                    priority=rec.get("priority", "medium"),
                    title=rec.get("title", ""),
                    description=rec.get("description", ""),
                    estimated_effort=rec.get("estimated_effort", "2-4 weeks"),
                    applicable_articles=rec.get("applicable_articles", []),
                )
            )

        critical_count = sum(1 for r in recommendations if r.priority == "critical")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"dd_recommendations:{dd_level.value}:{country_code}",
            str(len(recommendations)),
        )

        logger.info(
            "DD recommendations retrieved: level=%s country=%s total=%d critical=%d operator=%s",
            dd_level.value,
            country_code or "global",
            len(recommendations),
            critical_count,
            user.operator_id or user.user_id,
        )

        return DDRecommendationsResponse(
            country_code=country_code.upper() if country_code else None,
            dd_level=dd_level,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            critical_count=critical_count,
            implementation_timeline=result.get("implementation_timeline", "3-6 months"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["EUDR Article 29 Criteria", "Internal Best Practices"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DD recommendations retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DD recommendations retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/country-classifications
# ---------------------------------------------------------------------------


@router.get(
    "/country-classifications",
    response_model=CountryClassificationResponse,
    summary="Get EUDR country classifications",
    description=(
        "Retrieve the full list of EUDR country classifications based on "
        "corruption indices per Article 29. Shows each country's classification "
        "(low_risk, standard_risk, high_risk), required DD level, and underlying "
        "CPI/WGI scores. Supports pagination and region filtering."
    ),
    responses={
        200: {"description": "Country classifications retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_country_classifications(
    request: Request,
    classification: Optional[CountryClassificationEnum] = Query(
        None,
        description="Filter by classification level",
    ),
    region: Optional[RegionEnum] = Query(None, description="Filter by region"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CountryClassificationResponse:
    """Get EUDR country classifications.

    Args:
        classification: Optional classification level filter.
        region: Optional region filter.
        pagination: Pagination parameters.
        user: Authenticated user with compliance:read permission.

    Returns:
        CountryClassificationResponse with classified countries.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_engine()
        result = engine.get_country_classifications(
            classification=classification.value if classification else None,
            region=region.value if region else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        classifications = []
        for entry in result.get("classifications", []):
            country_class = CountryClassificationEnum(
                entry.get("classification", "standard_risk")
            )
            classifications.append(
                CountryClassificationEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    classification=country_class,
                    required_dd_level=_map_dd_level(country_class),
                    cpi_score=Decimal(str(entry.get("cpi_score", 0))) if entry.get("cpi_score") else None,
                    wgi_composite=Decimal(str(entry.get("wgi_composite", 0))) if entry.get("wgi_composite") else None,
                    risk_level=_classify_risk(country_class),
                    region=RegionEnum(entry.get("region")) if entry.get("region") else None,
                )
            )

        total = result.get("total_countries", len(classifications))
        low_count = result.get("low_risk_count", 0)
        standard_count = result.get("standard_risk_count", 0)
        high_count = result.get("high_risk_count", 0)
        classification_date_val = result.get("classification_date", date.today())

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"classifications:{classification}:{region}",
            str(total),
        )

        logger.info(
            "Country classifications retrieved: total=%d low=%d standard=%d high=%d operator=%s",
            total,
            low_count,
            standard_count,
            high_count,
            user.operator_id or user.user_id,
        )

        return CountryClassificationResponse(
            classifications=classifications,
            total_countries=total,
            low_risk_count=low_count,
            standard_risk_count=standard_count,
            high_risk_count=high_count,
            classification_date=classification_date_val,
            next_review_date=result.get("next_review_date"),
            methodology="CPI + WGI composite per EUDR Article 29",
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
                data_sources=[
                    "Transparency International CPI",
                    "World Bank WGI",
                    "EUDR Article 29 Criteria",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Country classifications retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Country classifications retrieval failed",
        )
