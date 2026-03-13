# -*- coding: utf-8 -*-
"""
Correlation Analysis Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for deforestation-corruption correlation analysis using Pearson
correlation, regression modeling, heatmap generation, and causal pathway
identification.

Endpoints:
    POST /correlation/analyze                       - Run correlation analysis
    GET  /correlation/{country_code}/deforestation  - Country deforestation link
    POST /correlation/regression                    - Build regression model
    GET  /correlation/heatmap                       - Corruption-deforestation heatmap
    GET  /correlation/causal-pathways               - Causal pathway analysis

Minimum 10 data points required for valid correlation analysis.
Significance level: p < 0.05 by default.
Data Sources: Transparency International CPI, World Bank WGI, Global Forest Watch

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Deforestation Correlation Engine
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
    get_correlation_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    CausalPathway,
    CausalPathwayResponse,
    CausalPathwayStep,
    CorrelationAnalysisRequest,
    CorrelationAnalysisResponse,
    CorrelationResultEntry,
    CorrelationStrengthEnum,
    DeforestationLinkResponse,
    ErrorResponse as SchemaErrorResponse,
    HeatmapCell,
    HeatmapResponse,
    MetadataSchema,
    ProvenanceInfo,
    RegressionRequest,
    RegressionResponse,
    RiskLevelEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/correlation", tags=["Deforestation Correlation"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_strength(r: Decimal) -> CorrelationStrengthEnum:
    """Classify correlation strength from Pearson r value."""
    abs_r = abs(r)
    if abs_r >= Decimal("0.8"):
        return CorrelationStrengthEnum.VERY_STRONG
    elif abs_r >= Decimal("0.6"):
        return CorrelationStrengthEnum.STRONG
    elif abs_r >= Decimal("0.4"):
        return CorrelationStrengthEnum.MODERATE
    elif abs_r >= Decimal("0.2"):
        return CorrelationStrengthEnum.WEAK
    return CorrelationStrengthEnum.NEGLIGIBLE


# ---------------------------------------------------------------------------
# POST /correlation/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=CorrelationAnalysisResponse,
    summary="Run corruption-deforestation correlation analysis",
    description=(
        "Perform statistical correlation analysis between corruption indices "
        "and deforestation rates using Pearson correlation with significance "
        "testing. Minimum 10 data points required for valid analysis."
    ),
    responses={
        200: {"description": "Correlation analysis completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_correlation(
    request: Request,
    body: CorrelationAnalysisRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:correlation:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CorrelationAnalysisResponse:
    """Run corruption-deforestation correlation analysis.

    Args:
        body: Correlation analysis request with parameters.
        user: Authenticated user with correlation:create permission.

    Returns:
        CorrelationAnalysisResponse with statistical results.
    """
    start = time.monotonic()

    try:
        engine = get_correlation_engine()
        result = engine.analyze_correlation(
            country_codes=body.country_codes,
            index_type=body.index_type,
            deforestation_metric=body.deforestation_metric,
            start_year=body.start_year,
            end_year=body.end_year,
            min_data_points=body.min_data_points,
        )

        correlations = []
        for cr in result.get("correlations", []):
            pearson_r = Decimal(str(cr.get("pearson_r", 0)))
            p_val = Decimal(str(cr.get("p_value", 1)))
            correlations.append(
                CorrelationResultEntry(
                    variable_pair=cr.get("variable_pair", ""),
                    pearson_r=pearson_r,
                    p_value=p_val,
                    significant=p_val < Decimal("0.05"),
                    strength=_classify_strength(pearson_r),
                    n_observations=cr.get("n_observations", 0),
                    direction="negative" if pearson_r < Decimal("0") else "positive" if pearson_r > Decimal("0") else "none",
                )
            )

        primary_data = result.get("primary_correlation", {})
        primary_r = Decimal(str(primary_data.get("pearson_r", 0)))
        primary_p = Decimal(str(primary_data.get("p_value", 1)))
        primary = CorrelationResultEntry(
            variable_pair=primary_data.get("variable_pair", "CPI vs Deforestation"),
            pearson_r=primary_r,
            p_value=primary_p,
            significant=primary_p < Decimal("0.05"),
            strength=_classify_strength(primary_r),
            n_observations=primary_data.get("n_observations", 0),
            direction="negative" if primary_r < Decimal("0") else "positive" if primary_r > Decimal("0") else "none",
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"correlation:{body.index_type}", str(primary_r)
        )

        logger.info(
            "Correlation analysis completed: r=%s p=%s countries=%d operator=%s",
            primary_r,
            primary_p,
            result.get("total_countries_analyzed", 0),
            user.operator_id or user.user_id,
        )

        return CorrelationAnalysisResponse(
            correlations=correlations,
            primary_correlation=primary,
            analysis_period=result.get("analysis_period", ""),
            total_countries_analyzed=result.get("total_countries_analyzed", 0),
            data_quality_score=Decimal(str(result.get("data_quality_score", 0.5))),
            interpretation=result.get("interpretation", ""),
            caveats=result.get("caveats", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "Global Forest Watch"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Correlation analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Correlation analysis failed",
        )


# ---------------------------------------------------------------------------
# GET /correlation/{country_code}/deforestation
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/deforestation",
    response_model=DeforestationLinkResponse,
    summary="Get corruption-deforestation link for a country",
    description=(
        "Retrieve country-specific analysis of the link between corruption "
        "levels and deforestation rates including risk amplification factor, "
        "historical comparison, and peer benchmarks."
    ),
    responses={
        200: {"description": "Deforestation link retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_deforestation_link(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:correlation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DeforestationLinkResponse:
    """Get corruption-deforestation link analysis for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with correlation:read permission.

    Returns:
        DeforestationLinkResponse with country-specific correlation data.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_correlation_engine()
        result = engine.get_deforestation_link(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deforestation link data not found for {normalized_code}",
            )

        corr_data = result.get("correlation", {})
        pearson_r = Decimal(str(corr_data.get("pearson_r", 0)))
        p_val = Decimal(str(corr_data.get("p_value", 1)))

        correlation = CorrelationResultEntry(
            variable_pair=corr_data.get("variable_pair", "CPI vs Forest Loss"),
            pearson_r=pearson_r,
            p_value=p_val,
            significant=p_val < Decimal("0.05"),
            strength=_classify_strength(pearson_r),
            n_observations=corr_data.get("n_observations", 0),
            direction="negative" if pearson_r < Decimal("0") else "positive" if pearson_r > Decimal("0") else "none",
        )

        historical = {}
        for k, v in result.get("historical_comparison", {}).items():
            historical[k] = Decimal(str(v))

        peer_benchmarks = {}
        for k, v in result.get("peer_benchmarks", {}).items():
            peer_benchmarks[k] = Decimal(str(v))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"deforestation_link:{normalized_code}", str(pearson_r)
        )

        logger.info(
            "Deforestation link retrieved: country=%s r=%s operator=%s",
            normalized_code,
            pearson_r,
            user.operator_id or user.user_id,
        )

        return DeforestationLinkResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            corruption_score=Decimal(str(result.get("corruption_score", 0))),
            deforestation_rate=Decimal(str(result.get("deforestation_rate", 0))),
            correlation=correlation,
            risk_amplification_factor=Decimal(str(result.get("risk_amplification_factor", 1.0))),
            historical_comparison=historical,
            peer_benchmarks=peer_benchmarks,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "Global Forest Watch"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Deforestation link retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deforestation link retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /correlation/regression
# ---------------------------------------------------------------------------


@router.post(
    "/regression",
    response_model=RegressionResponse,
    summary="Build regression model for corruption-deforestation link",
    description=(
        "Build a regression model to quantify the relationship between "
        "corruption indices and deforestation. Supports linear, polynomial, "
        "and logistic regression types."
    ),
    responses={
        200: {"description": "Regression model built"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def build_regression(
    request: Request,
    body: RegressionRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:correlation:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> RegressionResponse:
    """Build a regression model for corruption-deforestation analysis.

    Args:
        body: Regression request with variables and model type.
        user: Authenticated user with correlation:create permission.

    Returns:
        RegressionResponse with model coefficients and statistics.
    """
    start = time.monotonic()

    try:
        engine = get_correlation_engine()
        result = engine.build_regression(
            dependent_variable=body.dependent_variable,
            independent_variables=body.independent_variables,
            country_codes=body.country_codes,
            model_type=body.model_type,
        )

        coefficients = {}
        for k, v in result.get("coefficients", {}).items():
            coefficients[k] = Decimal(str(v))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"regression:{body.model_type}:{body.dependent_variable}",
            str(result.get("r_squared", 0)),
        )

        logger.info(
            "Regression model built: type=%s r2=%s n=%d operator=%s",
            body.model_type,
            result.get("r_squared", 0),
            result.get("n_observations", 0),
            user.operator_id or user.user_id,
        )

        return RegressionResponse(
            model_type=body.model_type,
            dependent_variable=body.dependent_variable,
            independent_variables=body.independent_variables,
            coefficients=coefficients,
            intercept=Decimal(str(result.get("intercept", 0))),
            r_squared=Decimal(str(result.get("r_squared", 0))),
            adjusted_r_squared=Decimal(str(result.get("adjusted_r_squared", 0))),
            f_statistic=Decimal(str(result.get("f_statistic", 0))) if result.get("f_statistic") else None,
            p_value=Decimal(str(result.get("p_value", 1))),
            n_observations=result.get("n_observations", 0),
            residual_std_error=Decimal(str(result.get("residual_std_error", 0))) if result.get("residual_std_error") else None,
            interpretation=result.get("interpretation", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "Global Forest Watch"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Regression model building failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Regression model building failed",
        )


# ---------------------------------------------------------------------------
# GET /correlation/heatmap
# ---------------------------------------------------------------------------


@router.get(
    "/heatmap",
    response_model=HeatmapResponse,
    summary="Get corruption-deforestation heatmap data",
    description=(
        "Generate heatmap data showing the relationship between corruption "
        "levels and deforestation rates across countries. Countries are "
        "classified into quadrants for risk visualization."
    ),
    responses={
        200: {"description": "Heatmap data retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_heatmap(
    request: Request,
    year: Optional[int] = Query(None, ge=2000, le=2030, description="Analysis year"),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:correlation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> HeatmapResponse:
    """Get corruption-deforestation heatmap data.

    Args:
        year: Analysis year (default: latest).
        user: Authenticated user with correlation:read permission.

    Returns:
        HeatmapResponse with heatmap cells and quadrant analysis.
    """
    start = time.monotonic()

    try:
        engine = get_correlation_engine()
        result = engine.get_heatmap(year=year)

        cells = []
        for cell_data in result.get("cells", []):
            corruption = Decimal(str(cell_data.get("corruption_score", 0)))
            deforestation = Decimal(str(cell_data.get("deforestation_rate", 0)))
            quadrant = cell_data.get("risk_quadrant", "low_corruption_low_deforestation")

            risk = RiskLevelEnum.LOW
            if "high_corruption_high_deforestation" in quadrant:
                risk = RiskLevelEnum.CRITICAL
            elif "high_corruption" in quadrant or "high_deforestation" in quadrant:
                risk = RiskLevelEnum.HIGH

            cells.append(
                HeatmapCell(
                    country_code=cell_data.get("country_code", ""),
                    country_name=cell_data.get("country_name", ""),
                    corruption_score=corruption,
                    deforestation_rate=deforestation,
                    risk_quadrant=quadrant,
                    risk_level=risk,
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"heatmap:{year}", str(len(cells))
        )

        logger.info(
            "Heatmap data retrieved: cells=%d year=%s operator=%s",
            len(cells),
            year,
            user.operator_id or user.user_id,
        )

        return HeatmapResponse(
            cells=cells,
            corruption_axis_label=result.get("corruption_axis_label", "CPI Score (0-100)"),
            deforestation_axis_label=result.get("deforestation_axis_label", "Annual Forest Loss (hectares)"),
            quadrant_counts=result.get("quadrant_counts", {}),
            total_countries=len(cells),
            high_risk_quadrant_countries=result.get("high_risk_quadrant_countries", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "Global Forest Watch"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Heatmap retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Heatmap data retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /correlation/causal-pathways
# ---------------------------------------------------------------------------


@router.get(
    "/causal-pathways",
    response_model=CausalPathwayResponse,
    summary="Get causal pathway analysis for corruption-deforestation link",
    description=(
        "Identify and analyze causal pathways through which corruption "
        "leads to increased deforestation. Includes mechanism descriptions, "
        "evidence strength, and EUDR relevance scoring."
    ),
    responses={
        200: {"description": "Causal pathways retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_causal_pathways(
    request: Request,
    country_code: Optional[str] = Query(None, description="Optional country filter"),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:correlation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CausalPathwayResponse:
    """Get causal pathway analysis for corruption-deforestation link.

    Args:
        country_code: Optional country filter for context-specific pathways.
        user: Authenticated user with correlation:read permission.

    Returns:
        CausalPathwayResponse with identified causal pathways.
    """
    start = time.monotonic()

    try:
        engine = get_correlation_engine()
        result = engine.get_causal_pathways(
            country_code=country_code.upper() if country_code else None,
        )

        pathways = []
        for pw in result.get("pathways", []):
            steps = []
            for step_data in pw.get("steps", []):
                steps.append(
                    CausalPathwayStep(
                        step_number=step_data.get("step_number", 1),
                        mechanism=step_data.get("mechanism", ""),
                        evidence_strength=CorrelationStrengthEnum(step_data.get("evidence_strength", "moderate")),
                        supporting_data=step_data.get("supporting_data", []),
                    )
                )
            pathways.append(
                CausalPathway(
                    pathway_id=pw.get("pathway_id", ""),
                    pathway_name=pw.get("pathway_name", ""),
                    steps=steps,
                    overall_evidence_strength=CorrelationStrengthEnum(pw.get("overall_evidence_strength", "moderate")),
                    relevance_to_eudr=Decimal(str(pw.get("relevance_to_eudr", 0.5))),
                )
            )

        primary = result.get("primary_pathway", pathways[0].pathway_id if pathways else "")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"causal_pathways:{country_code}", str(len(pathways))
        )

        logger.info(
            "Causal pathways retrieved: pathways=%d country=%s operator=%s",
            len(pathways),
            country_code or "global",
            user.operator_id or user.user_id,
        )

        return CausalPathwayResponse(
            pathways=pathways,
            primary_pathway=primary,
            total_pathways=len(pathways),
            methodology_notes=result.get("methodology_notes", ""),
            limitations=result.get("limitations", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Academic Literature", "Global Forest Watch", "Internal Analysis"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Causal pathways retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Causal pathway analysis failed",
        )
