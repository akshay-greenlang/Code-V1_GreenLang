# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-028: Risk Assessment Engine

REST API endpoints for EUDR risk assessment operations. Provides 12
endpoints for full pipeline execution, individual engine delegation,
manual overrides, trend analysis, and health monitoring.

Endpoint Summary (12):
    POST /assess                          - Full risk assessment pipeline
    GET  /assess/{operation_id}           - Get assessment operation status
    POST /composite-score                 - Calculate composite risk score
    POST /evaluate-criteria               - Evaluate Article 10(2) criteria
    GET  /benchmarks/{country_code}       - Get country benchmark
    POST /benchmarks/batch                - Batch country benchmarks
    POST /classify                        - Classify risk level
    POST /simplified-dd/check             - Check simplified DD eligibility
    POST /override                        - Apply risk override
    GET  /trend/{operator_id}/{commodity} - Get risk trend
    POST /assess/batch                    - Batch risk assessment
    GET  /health                          - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-risk-assessment-engine:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 (GL-EUDR-RAE-028)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    CompositeRiskScore,
    CountryBenchmark,
    EUDRCommodity,
    OverrideReason,
    RiskAssessmentOperation,
    RiskAssessmentReport,
    RiskFactorInput,
    RiskLevel,
    RiskOverride,
    RiskTrendAnalysis,
    SimplifiedDDEligibility,
)
from greenlang.agents.eudr.risk_assessment_engine.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class AssessRiskRequest(GreenLangBase):
    """Request body for the full risk assessment pipeline."""

    operator_id: str = Field(..., description="EUDR operator identifier")
    commodity: EUDRCommodity = Field(..., description="EUDR regulated commodity")
    country_codes: List[str] = Field(
        ..., description="ISO 3166-1 alpha-2 country codes for sourcing origins"
    )
    supplier_ids: List[str] = Field(
        default_factory=list,
        description="Supplier identifiers in the supply chain",
    )


class CalculateCompositeRequest(GreenLangBase):
    """Request body for composite risk score calculation."""

    factor_inputs: List[Dict[str, Any]] = Field(
        ..., description="Risk factor input data"
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes for weighting",
    )


class EvaluateCriteriaRequest(GreenLangBase):
    """Request body for Article 10(2) criteria evaluation."""

    factor_inputs: List[Dict[str, Any]] = Field(
        ..., description="Risk factor input data"
    )
    composite_score: Dict[str, Any] = Field(
        ..., description="Composite risk score data"
    )
    country_codes: List[str] = Field(
        ..., description="ISO 3166-1 alpha-2 country codes for benchmarks"
    )


class ApplyOverrideRequest(GreenLangBase):
    """Request body for applying a risk override."""

    assessment_id: str = Field(
        ..., description="Risk assessment operation ID to override"
    )
    overridden_score: Decimal = Field(
        ..., ge=0, le=1, description="New score value (0.00 - 1.00)"
    )
    reason: OverrideReason = Field(
        ..., description="Reason category for the override"
    )
    justification: str = Field(
        ..., min_length=10, description="Detailed justification text"
    )
    overridden_by: str = Field(
        ..., description="User or system identifier performing override"
    )


class BatchAssessRequest(GreenLangBase):
    """Request body for batch risk assessment."""

    assessments: List[AssessRiskRequest] = Field(
        ..., description="List of risk assessment requests"
    )


class ErrorResponse(GreenLangBase):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field("internal_error", description="Error classification")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/risk-assessment-engine",
    tags=["EUDR Risk Assessment Engine"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=RiskAssessmentOperation,
    status_code=200,
    summary="Execute full risk assessment pipeline",
    description=(
        "Executes the complete 10-step risk assessment pipeline for a "
        "given operator, commodity, and sourcing countries: aggregate "
        "risk factors, calculate composite score, evaluate Article 10(2) "
        "criteria, get country benchmarks, classify risk level, check "
        "simplified DD eligibility, record trend data, and generate "
        "risk assessment report."
    ),
)
async def assess_risk(
    request: AssessRiskRequest,
) -> RiskAssessmentOperation:
    """Start a full risk assessment operation.

    Args:
        request: Assessment request with operator, commodity,
                countries, and suppliers.

    Returns:
        RiskAssessmentOperation with status, scores, and report reference.
    """
    try:
        service = get_service()
        operation = await service.assess_risk(
            operator_id=request.operator_id,
            commodity=request.commodity.value,
            country_codes=request.country_codes,
            supplier_ids=request.supplier_ids,
        )
        return operation
    except Exception as e:
        logger.error(
            f"assess_risk failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)[:200]}",
        )


@router.get(
    "/assess/{operation_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get risk assessment operation status",
    description=(
        "Retrieve the current status of a risk assessment operation "
        "by its operation identifier."
    ),
)
async def get_assessment_status(
    operation_id: str,
) -> Dict[str, Any]:
    """Get the status of a risk assessment operation.

    Args:
        operation_id: Operation identifier.

    Returns:
        Operation status details.
    """
    try:
        # In production, this queries the database for the operation.
        # For now, return a placeholder indicating the operation lookup.
        return {
            "operation_id": operation_id,
            "message": "Operation lookup requires database persistence layer",
            "status": "lookup_pending",
        }
    except Exception as e:
        logger.error(
            f"get_assessment_status failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Status lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/composite-score",
    response_model=CompositeRiskScore,
    status_code=200,
    summary="Calculate composite risk score",
    description=(
        "Calculate a composite risk score from individual risk factor "
        "inputs using multi-dimensional weighted scoring across "
        "environmental, governance, supply chain, and social dimensions."
    ),
)
async def calculate_composite_score(
    request: CalculateCompositeRequest,
) -> CompositeRiskScore:
    """Calculate a composite risk score.

    Args:
        request: Factor inputs and optional country codes for weighting.

    Returns:
        CompositeRiskScore with overall score and dimension breakdowns.
    """
    try:
        service = get_service()

        # Convert dict inputs to RiskFactorInput models
        factor_inputs = [
            RiskFactorInput(**fi) for fi in request.factor_inputs
        ]

        score = await service.calculate_composite_score(
            factor_inputs=factor_inputs,
            country_codes=request.country_codes,
        )
        return score
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"calculate_composite_score failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Composite score calculation failed: {str(e)[:200]}",
        )


@router.post(
    "/evaluate-criteria",
    response_model=Article10CriteriaResult,
    status_code=200,
    summary="Evaluate Article 10(2) criteria",
    description=(
        "Evaluate the 7 criteria specified in EUDR Article 10(2) for "
        "risk assessment: (a) deforestation prevalence, (b) conflict/"
        "sanctions, (c) corruption index, (d) legal complexity, "
        "(e) supply chain complexity, (f) mixing/substitution risk, "
        "(g) indigenous rights concerns."
    ),
)
async def evaluate_criteria(
    request: EvaluateCriteriaRequest,
) -> Article10CriteriaResult:
    """Evaluate Article 10(2) criteria.

    Args:
        request: Factor inputs, composite score, and country codes.

    Returns:
        Article10CriteriaResult with per-criterion evaluation.
    """
    try:
        service = get_service()

        factor_inputs = [
            RiskFactorInput(**fi) for fi in request.factor_inputs
        ]

        # Get benchmarks for the countries
        benchmarks = await service.get_country_benchmarks(
            country_codes=request.country_codes
        )

        # Reconstruct composite score
        composite = CompositeRiskScore(**request.composite_score)

        result = await service.evaluate_article10_criteria(
            factor_inputs=factor_inputs,
            benchmarks=benchmarks,
            composite=composite,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"evaluate_criteria failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Article 10 criteria evaluation failed: {str(e)[:200]}",
        )


@router.get(
    "/benchmarks/{country_code}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get country benchmark",
    description=(
        "Retrieve the EU-published benchmark classification for a "
        "specific country (low/standard/high risk) per EUDR Article 29."
    ),
)
async def get_country_benchmark(
    country_code: str,
) -> Dict[str, Any]:
    """Get the country benchmark for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Country benchmark data including risk level and indicators.
    """
    try:
        service = get_service()
        benchmarks = await service.get_country_benchmarks(
            country_codes=[country_code.upper()]
        )

        if not benchmarks:
            raise HTTPException(
                status_code=404,
                detail=f"No benchmark found for country: {country_code}",
            )

        return benchmarks[0].model_dump(mode="json")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"get_country_benchmark failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Country benchmark lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/benchmarks/batch",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="Batch country benchmarks",
    description=(
        "Retrieve EU-published benchmark classifications for multiple "
        "countries in a single request."
    ),
)
async def batch_country_benchmarks(
    country_codes: List[str],
) -> List[Dict[str, Any]]:
    """Get country benchmarks for multiple countries.

    Args:
        country_codes: List of ISO 3166-1 alpha-2 country codes.

    Returns:
        List of country benchmark data dictionaries.
    """
    try:
        service = get_service()
        benchmarks = await service.get_country_benchmarks(
            country_codes=[c.upper() for c in country_codes]
        )
        return [b.model_dump(mode="json") for b in benchmarks]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"batch_country_benchmarks failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch benchmark lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/classify",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Classify risk level",
    description=(
        "Classify the risk level (negligible/low/standard/high/critical) "
        "from a composite risk score and optional Article 10(2) result."
    ),
)
async def classify_risk(
    composite_score: Dict[str, Any],
    article10_result: Optional[Dict[str, Any]] = None,
    previous_level: Optional[RiskLevel] = None,
) -> Dict[str, Any]:
    """Classify risk level from composite score.

    Args:
        composite_score: Composite risk score data.
        article10_result: Optional Article 10(2) evaluation result.
        previous_level: Optional previous risk level for stability.

    Returns:
        Classification result with risk level and confidence.
    """
    try:
        service = get_service()

        composite = CompositeRiskScore(**composite_score)
        article10 = (
            Article10CriteriaResult(**article10_result)
            if article10_result
            else None
        )

        level = await service.classify_risk(
            composite=composite,
            article10=article10,
            previous=previous_level,
        )

        return {
            "risk_level": level.value,
            "composite_score": str(composite.overall_score),
            "classification_basis": "article10_enhanced" if article10 else "composite_only",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"classify_risk failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk classification failed: {str(e)[:200]}",
        )


@router.post(
    "/simplified-dd/check",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Check simplified due diligence eligibility",
    description=(
        "Check whether an operator qualifies for simplified due "
        "diligence per EUDR Article 13, based on composite risk "
        "score and country benchmark classifications."
    ),
)
async def check_simplified_dd(
    composite_score: Dict[str, Any],
    country_codes: List[str],
) -> Dict[str, Any]:
    """Check simplified DD eligibility.

    Args:
        composite_score: Composite risk score data.
        country_codes: ISO 3166-1 alpha-2 country codes.

    Returns:
        Eligibility determination with reasoning.
    """
    try:
        service = get_service()

        composite = CompositeRiskScore(**composite_score)
        benchmarks = await service.get_country_benchmarks(
            country_codes=[c.upper() for c in country_codes]
        )

        eligibility = await service.check_simplified_dd(
            composite=composite,
            benchmarks=benchmarks,
        )

        return eligibility.model_dump(mode="json")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"check_simplified_dd failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Simplified DD check failed: {str(e)[:200]}",
        )


@router.post(
    "/override",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Apply risk override",
    description=(
        "Apply a manual risk score override to a completed assessment. "
        "Overrides require justification and are recorded in the "
        "provenance audit trail per EUDR Article 31."
    ),
)
async def apply_override(
    request: ApplyOverrideRequest,
) -> Dict[str, Any]:
    """Apply a manual risk override.

    Args:
        request: Override parameters including score, reason,
                and justification.

    Returns:
        RiskOverride data with provenance hash.
    """
    try:
        service = get_service()
        override = await service.apply_override(
            assessment_id=request.assessment_id,
            overridden_score=request.overridden_score,
            reason=request.reason,
            justification=request.justification,
            overridden_by=request.overridden_by,
        )
        return override.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"apply_override failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk override failed: {str(e)[:200]}",
        )


@router.get(
    "/trend/{operator_id}/{commodity}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get risk trend analysis",
    description=(
        "Retrieve temporal risk trend analysis for a specific operator "
        "and commodity, including trend direction, drift detection, "
        "and historical data points."
    ),
)
async def get_risk_trend(
    operator_id: str,
    commodity: str,
) -> Dict[str, Any]:
    """Get risk trend for an operator and commodity.

    Args:
        operator_id: EUDR operator identifier.
        commodity: Commodity being assessed.

    Returns:
        Risk trend analysis with direction and historical data.
    """
    try:
        service = get_service()
        trend = await service.get_risk_trend(
            operator_id=operator_id,
            commodity=commodity,
        )
        return trend.model_dump(mode="json")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"get_risk_trend failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk trend lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/assess/batch",
    response_model=List[RiskAssessmentOperation],
    status_code=200,
    summary="Batch risk assessment",
    description=(
        "Execute multiple risk assessments in batch. Each assessment "
        "is processed sequentially to maintain deterministic ordering "
        "and provenance chain integrity."
    ),
)
async def batch_assess_risk(
    request: BatchAssessRequest,
) -> List[RiskAssessmentOperation]:
    """Execute batch risk assessments.

    Args:
        request: List of assessment requests.

    Returns:
        List of RiskAssessmentOperation results.
    """
    try:
        service = get_service()

        # Convert request models to dicts for the service layer
        assessment_dicts = [
            {
                "operator_id": a.operator_id,
                "commodity": a.commodity.value,
                "country_codes": a.country_codes,
                "supplier_ids": a.supplier_ids,
            }
            for a in request.assessments
        ]

        results = await service.batch_assess_risk(requests=assessment_dicts)
        return results
    except Exception as e:
        logger.error(
            f"batch_assess_risk failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch risk assessment failed: {str(e)[:200]}",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Health check",
    description=(
        "Returns the health status of the Risk Assessment Engine "
        "including engine availability, database connectivity, and "
        "Redis connectivity."
    ),
)
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Risk Assessment Engine.

    Returns:
        Dictionary with component health statuses.
    """
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error(
            f"health_check failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        return {
            "agent_id": "GL-EUDR-RAE-028",
            "status": "error",
            "error": str(e)[:200],
        }


def get_router() -> APIRouter:
    """Return the Risk Assessment Engine API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
