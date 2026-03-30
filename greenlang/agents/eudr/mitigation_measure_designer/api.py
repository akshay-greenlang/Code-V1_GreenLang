# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-029: Mitigation Measure Designer

REST API endpoints for EUDR mitigation measure design operations.
Provides 12 endpoints for strategy design, measure lifecycle
management, template browsing, risk reduction verification, report
generation, workflow orchestration, and health monitoring.

Endpoint Summary (12):
    POST /design-strategy                     - Design mitigation strategy
    GET  /strategies/{strategy_id}            - Get strategy details
    GET  /strategies                          - List strategies with filters
    POST /measures/{measure_id}/approve       - Approve a measure
    POST /measures/{measure_id}/start         - Start measure implementation
    POST /measures/{measure_id}/complete      - Complete a measure
    POST /verify/{strategy_id}               - Verify risk reduction
    GET  /templates                           - List measure templates
    GET  /templates/{template_id}            - Get template details
    POST /generate-report/{strategy_id}      - Generate mitigation report
    GET  /workflows/{workflow_id}/status      - Get workflow status
    GET  /health                              - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-mitigation-measure-designer:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 13, 14-16, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.mitigation_measure_designer.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class DesignStrategyRequest(GreenLangBase):
    """Request body for designing a mitigation strategy."""

    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    commodity: str = Field(
        ..., description="EUDR regulated commodity (e.g. 'cocoa', 'palm_oil')"
    )
    risk_level: str = Field(
        ..., description="Risk level (negligible/low/standard/high/critical)"
    )
    risk_score: str = Field(
        ..., description="Current risk score (0-100 as string)"
    )
    risk_dimension: str = Field(
        ...,
        description=(
            "Risk dimension (country/supplier/commodity/"
            "corruption/deforestation/environmental)"
        ),
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes for sourcing origins",
    )
    supplier_ids: List[str] = Field(
        default_factory=list,
        description="Supplier identifiers in the supply chain",
    )
    trigger_source: str = Field(
        default="manual",
        description="Source that triggered the mitigation (e.g. 'eudr-028', 'manual')",
    )


class CompleteMeasureRequest(GreenLangBase):
    """Request body for completing a measure."""

    actual_reduction: Optional[str] = Field(
        None,
        description=(
            "Actual risk reduction achieved (optional, decimal string). "
            "If omitted, estimated reduction is used."
        ),
    )


class CancelMeasureRequest(GreenLangBase):
    """Request body for cancelling a measure."""

    reason: str = Field(
        ...,
        min_length=5,
        description="Cancellation reason (minimum 5 characters)",
    )


class ApproveMeasureRequest(GreenLangBase):
    """Request body for approving a measure."""

    approved_by: str = Field(
        ..., description="User or system identifier approving the measure"
    )


class AddEvidenceRequest(GreenLangBase):
    """Request body for adding evidence to a measure."""

    evidence_type: str = Field(
        ..., description="Type of evidence (certificate/report/photo/document/audit)"
    )
    title: str = Field(
        ..., min_length=3, description="Human-readable evidence title"
    )
    file_ref: str = Field(
        ..., description="File reference, storage path, or URL"
    )
    uploaded_by: str = Field(
        ..., description="User who uploaded the evidence"
    )


class InitiateWorkflowRequest(GreenLangBase):
    """Request body for initiating a mitigation workflow."""

    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    commodity: str = Field(
        ..., description="EUDR regulated commodity"
    )
    risk_level: str = Field(
        ..., description="Risk level triggering the workflow"
    )
    risk_score: str = Field(
        ..., description="Current risk score (0-100 as string)"
    )
    risk_dimension: str = Field(
        ..., description="Risk dimension"
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes",
    )


class ErrorResponse(GreenLangBase):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field(
        "internal_error", description="Error classification"
    )
    timestamp: Optional[str] = Field(
        None, description="Error timestamp"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/mitigation-measure-designer",
    tags=["EUDR Mitigation Measure Designer"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)


# ---------------------------------------------------------------------------
# Helper: build RiskTrigger from request
# ---------------------------------------------------------------------------


def _build_risk_trigger(request: DesignStrategyRequest) -> Any:
    """Build a RiskTrigger-like object from a design strategy request.

    Attempts to import and instantiate the RiskTrigger model from the
    models module. Falls back to a SimpleNamespace if models are not
    available.

    Args:
        request: DesignStrategyRequest with risk parameters.

    Returns:
        RiskTrigger model or SimpleNamespace with equivalent attributes.
    """
    try:
        from greenlang.agents.eudr.mitigation_measure_designer.models import (
            EUDRCommodity,
            RiskDimension,
            RiskLevel,
            RiskTrigger,
        )

        return RiskTrigger(
            operator_id=request.operator_id,
            commodity=EUDRCommodity(request.commodity),
            risk_level=RiskLevel(request.risk_level),
            risk_score=Decimal(request.risk_score),
            risk_dimension=RiskDimension(request.risk_dimension),
            country_codes=request.country_codes,
            supplier_ids=request.supplier_ids,
            trigger_source=request.trigger_source,
        )
    except (ImportError, ValueError) as exc:
        # Fallback: use a SimpleNamespace
        from types import SimpleNamespace

        class _FakeEnum:
            """Minimal enum-like object for fallback."""

            def __init__(self, v: str) -> None:
                self.value = v

        return SimpleNamespace(
            operator_id=request.operator_id,
            commodity=_FakeEnum(request.commodity),
            risk_level=_FakeEnum(request.risk_level),
            risk_score=Decimal(request.risk_score),
            risk_dimension=_FakeEnum(request.risk_dimension),
            country_codes=request.country_codes,
            supplier_ids=request.supplier_ids,
            trigger_source=request.trigger_source,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/design-strategy",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Design a mitigation strategy",
    description=(
        "Analyse a risk trigger and design a structured mitigation strategy "
        "with prioritized measures per EUDR Article 11. The strategy "
        "includes measure selection from the template library, "
        "three-scenario effectiveness estimation, and target risk "
        "reduction calculation."
    ),
)
async def design_strategy(
    request: DesignStrategyRequest,
) -> Dict[str, Any]:
    """Design a mitigation strategy from a risk trigger.

    Args:
        request: Risk trigger details including operator, commodity,
                risk level, score, and dimension.

    Returns:
        MitigationStrategy data with measures, estimates, and provenance.
    """
    try:
        service = get_service()
        risk_trigger = _build_risk_trigger(request)
        strategy = await service.design_strategy(risk_trigger)

        if isinstance(strategy, dict):
            return strategy
        return strategy.model_dump(mode="json") if hasattr(strategy, "model_dump") else dict(strategy)

    except ValueError as e:
        logger.warning(
            f"design_strategy validation error: {str(e)[:500]}"
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"design_strategy failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Strategy design failed: {str(e)[:200]}",
        )


@router.get(
    "/strategies/{strategy_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get strategy details",
    description=(
        "Retrieve the full details of a mitigation strategy by its "
        "identifier, including all associated measures, effectiveness "
        "estimates, and provenance data."
    ),
)
async def get_strategy(
    strategy_id: str,
) -> Dict[str, Any]:
    """Get a mitigation strategy by identifier.

    Args:
        strategy_id: Strategy identifier.

    Returns:
        Strategy data dictionary.
    """
    try:
        service = get_service()
        strategy = await service.get_strategy(strategy_id)

        if strategy is None:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy {strategy_id} not found",
            )

        if isinstance(strategy, dict):
            return strategy
        return strategy.model_dump(mode="json") if hasattr(strategy, "model_dump") else dict(strategy)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"get_strategy failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Strategy lookup failed: {str(e)[:200]}",
        )


@router.get(
    "/strategies",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="List strategies with filters",
    description=(
        "List mitigation strategies with optional filters by operator, "
        "commodity, and status. Returns a list of strategy summaries."
    ),
)
async def list_strategies(
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    commodity: Optional[str] = Query(
        None, description="Filter by EUDR commodity"
    ),
    status: Optional[str] = Query(
        None, description="Filter by strategy status"
    ),
) -> List[Dict[str, Any]]:
    """List strategies with optional filtering.

    Args:
        operator_id: Optional operator ID filter.
        commodity: Optional commodity filter.
        status: Optional status filter.

    Returns:
        List of strategy data dictionaries.
    """
    try:
        service = get_service()
        strategies = await service.list_strategies(
            operator_id=operator_id,
            commodity=commodity,
            status=status,
        )

        results = []
        for s in strategies:
            if isinstance(s, dict):
                results.append(s)
            elif hasattr(s, "model_dump"):
                results.append(s.model_dump(mode="json"))
            else:
                results.append(dict(s))

        return results

    except Exception as e:
        logger.error(
            f"list_strategies failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Strategy listing failed: {str(e)[:200]}",
        )


@router.post(
    "/measures/{measure_id}/approve",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Approve a proposed measure",
    description=(
        "Approve a proposed mitigation measure for implementation. "
        "Transitions the measure from 'proposed' to 'approved' status "
        "and records the approver for audit trail."
    ),
)
async def approve_measure(
    measure_id: str,
    request: ApproveMeasureRequest,
) -> Dict[str, Any]:
    """Approve a proposed measure.

    Args:
        measure_id: Measure identifier.
        request: Approval request with approver identity.

    Returns:
        Updated measure data.
    """
    try:
        service = get_service()
        measure = await service.approve_measure(
            measure_id=measure_id,
            approved_by=request.approved_by,
        )

        if isinstance(measure, dict):
            return measure
        return measure.model_dump(mode="json") if hasattr(measure, "model_dump") else dict(measure)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"approve_measure failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Measure approval failed: {str(e)[:200]}",
        )


@router.post(
    "/measures/{measure_id}/start",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Start measure implementation",
    description=(
        "Begin implementation of an approved mitigation measure. "
        "Transitions the measure from 'approved' to 'in_progress' status."
    ),
)
async def start_measure(
    measure_id: str,
) -> Dict[str, Any]:
    """Start implementation of an approved measure.

    Args:
        measure_id: Measure identifier.

    Returns:
        Updated measure data.
    """
    try:
        service = get_service()
        measure = await service.start_measure(measure_id=measure_id)

        if isinstance(measure, dict):
            return measure
        return measure.model_dump(mode="json") if hasattr(measure, "model_dump") else dict(measure)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"start_measure failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Measure start failed: {str(e)[:200]}",
        )


@router.post(
    "/measures/{measure_id}/complete",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Complete a measure",
    description=(
        "Mark a measure as completed and optionally record the actual "
        "risk reduction achieved. Transitions the measure from "
        "'in_progress' to 'completed' status."
    ),
)
async def complete_measure(
    measure_id: str,
    request: Optional[CompleteMeasureRequest] = None,
) -> Dict[str, Any]:
    """Complete a measure.

    Args:
        measure_id: Measure identifier.
        request: Optional request body with actual reduction value.

    Returns:
        Updated measure data.
    """
    try:
        service = get_service()

        actual_reduction: Optional[Decimal] = None
        if request and request.actual_reduction:
            actual_reduction = Decimal(request.actual_reduction)

        measure = await service.complete_measure(
            measure_id=measure_id,
            actual_reduction=actual_reduction,
        )

        if isinstance(measure, dict):
            return measure
        return measure.model_dump(mode="json") if hasattr(measure, "model_dump") else dict(measure)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"complete_measure failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Measure completion failed: {str(e)[:200]}",
        )


@router.post(
    "/verify/{strategy_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Verify risk reduction",
    description=(
        "Verify risk reduction for a mitigation strategy by comparing "
        "the current risk score against the initial score at strategy "
        "creation time. Evaluates measure effectiveness and produces "
        "a verification report with gap analysis."
    ),
)
async def verify_risk_reduction(
    strategy_id: str,
) -> Dict[str, Any]:
    """Verify risk reduction for a strategy.

    Args:
        strategy_id: Strategy identifier.

    Returns:
        Verification report data.
    """
    try:
        service = get_service()
        report = await service.verify_risk_reduction(
            strategy_id=strategy_id,
        )

        if isinstance(report, dict):
            return report
        return report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"verify_risk_reduction failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk reduction verification failed: {str(e)[:200]}",
        )


@router.get(
    "/templates",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="List measure templates",
    description=(
        "List available mitigation measure templates with optional "
        "filters by risk dimension, Article 11 category, and EUDR "
        "commodity. Templates provide proven measure blueprints for "
        "strategy design."
    ),
)
async def list_templates(
    dimension: Optional[str] = Query(
        None, description="Filter by risk dimension"
    ),
    category: Optional[str] = Query(
        None, description="Filter by Article 11 category"
    ),
    commodity: Optional[str] = Query(
        None, description="Filter by EUDR commodity"
    ),
) -> List[Dict[str, Any]]:
    """List measure templates with optional filtering.

    Args:
        dimension: Optional risk dimension filter.
        category: Optional Article 11 category filter.
        commodity: Optional commodity filter.

    Returns:
        List of template data dictionaries.
    """
    try:
        service = get_service()
        templates = await service.list_templates(
            dimension=dimension,
            category=category,
            commodity=commodity,
        )

        results = []
        for t in templates:
            if isinstance(t, dict):
                results.append(t)
            elif hasattr(t, "model_dump"):
                results.append(t.model_dump(mode="json"))
            else:
                results.append(dict(t))

        return results

    except Exception as e:
        logger.error(
            f"list_templates failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Template listing failed: {str(e)[:200]}",
        )


@router.get(
    "/templates/{template_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get template details",
    description=(
        "Retrieve the full details of a mitigation measure template "
        "by its identifier, including applicability rules, expected "
        "effectiveness, and implementation guidance."
    ),
)
async def get_template(
    template_id: str,
) -> Dict[str, Any]:
    """Get a measure template by identifier.

    Args:
        template_id: Template identifier.

    Returns:
        Template data dictionary.
    """
    try:
        service = get_service()
        template = await service.get_template(template_id)

        if template is None:
            raise HTTPException(
                status_code=404,
                detail=f"Template {template_id} not found",
            )

        if isinstance(template, dict):
            return template
        return template.model_dump(mode="json") if hasattr(template, "model_dump") else dict(template)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"get_template failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Template lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/generate-report/{strategy_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Generate mitigation report",
    description=(
        "Generate a structured mitigation report for a strategy, "
        "suitable for inclusion in Due Diligence Statements (DDS) "
        "per EUDR Article 12(2). Includes measure summaries, "
        "effectiveness data, evidence references, and provenance hashes."
    ),
)
async def generate_report(
    strategy_id: str,
) -> Dict[str, Any]:
    """Generate a mitigation report for a strategy.

    Args:
        strategy_id: Strategy identifier.

    Returns:
        Mitigation report data.
    """
    try:
        service = get_service()
        report = await service.generate_report(
            strategy_id=strategy_id,
        )

        if isinstance(report, dict):
            return report
        return report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"generate_report failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)[:200]}",
        )


@router.get(
    "/workflows/{workflow_id}/status",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get workflow status",
    description=(
        "Retrieve the current status of a mitigation workflow, "
        "including phases completed, phases remaining, and "
        "associated strategy identifier."
    ),
)
async def get_workflow_status(
    workflow_id: str,
) -> Dict[str, Any]:
    """Get the current status of a mitigation workflow.

    Args:
        workflow_id: Workflow identifier.

    Returns:
        Workflow state data.
    """
    try:
        service = get_service()
        status = await service.get_workflow_status(
            workflow_id=workflow_id,
        )

        if isinstance(status, dict):
            return status
        return status.model_dump(mode="json") if hasattr(status, "model_dump") else dict(status)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"get_workflow_status failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Workflow status lookup failed: {str(e)[:200]}",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Health check",
    description=(
        "Returns the health status of the Mitigation Measure Designer "
        "including engine availability, database connectivity, Redis "
        "connectivity, and in-memory store statistics."
    ),
)
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Mitigation Measure Designer.

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
            "agent_id": "GL-EUDR-MMD-029",
            "status": "error",
            "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Mitigation Measure Designer API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
