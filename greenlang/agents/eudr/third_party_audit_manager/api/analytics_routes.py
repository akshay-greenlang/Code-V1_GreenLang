# -*- coding: utf-8 -*-
"""
Analytics Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for audit analytics including finding trends, auditor
performance benchmarking, compliance rates, CAR lifecycle analytics,
and executive dashboard aggregation.

Endpoints (5):
    GET /analytics/findings            - Finding trend analytics
    GET /analytics/auditor-performance - Auditor benchmarking
    GET /analytics/compliance-rates    - Compliance rate trends
    GET /analytics/car-performance     - CAR lifecycle analytics
    GET /analytics/dashboard           - Executive dashboard data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024, AuditAnalyticsEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    get_analytics_engine,
    rate_limit_standard,
    require_permission,
    validate_date_range,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    AuditorPerformanceResponse,
    CARPerformanceResponse,
    ComplianceRatesResponse,
    DashboardResponse,
    ErrorResponse,
    FindingTrendsResponse,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


@router.get(
    "/findings",
    response_model=FindingTrendsResponse,
    summary="Finding trend analytics",
    description="Retrieve NC finding trends by severity, country, and commodity.",
    responses={200: {"description": "Finding trends retrieved"}},
)
async def get_finding_trends(
    request: Request,
    country_code: Optional[str] = Query(None, description="Filter by country"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    period: Optional[str] = Query(default="monthly", description="Aggregation period"),
    date_range: Dict = Depends(validate_date_range),
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> FindingTrendsResponse:
    """Retrieve finding trend analytics.

    Args:
        country_code: Optional country filter.
        commodity: Optional commodity filter.
        period: Aggregation period (monthly/quarterly).
        date_range: Date range filter.
        user: Authenticated user with analytics:read permission.

    Returns:
        FindingTrendsResponse with trend data.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_finding_trends(
            country_code=country_code,
            commodity=commodity,
            period=period,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
        )
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return FindingTrendsResponse(
            trends=result.get("trends", []),
            total_findings=result.get("total_findings", 0),
            period_range=result.get("period_range", ""),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("analytics_findings", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Finding trends failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve finding trends",
        )


@router.get(
    "/auditor-performance",
    response_model=AuditorPerformanceResponse,
    summary="Auditor performance benchmarking",
    description="Retrieve auditor performance benchmarking data.",
    responses={200: {"description": "Performance data retrieved"}},
)
async def get_auditor_performance(
    request: Request,
    date_range: Dict = Depends(validate_date_range),
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> AuditorPerformanceResponse:
    """Retrieve auditor performance analytics.

    Args:
        date_range: Date range filter.
        user: Authenticated user with analytics:read permission.

    Returns:
        AuditorPerformanceResponse with benchmarking data.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_auditor_performance(
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
        )
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return AuditorPerformanceResponse(
            auditors=result.get("auditors", []),
            total_auditors=result.get("total_auditors", 0),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("analytics_auditor_perf", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auditor performance failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve auditor performance",
        )


@router.get(
    "/compliance-rates",
    response_model=ComplianceRatesResponse,
    summary="Compliance rate trends",
    description="Retrieve compliance rate trends over time.",
    responses={200: {"description": "Compliance rates retrieved"}},
)
async def get_compliance_rates(
    request: Request,
    period: Optional[str] = Query(default="monthly", description="Aggregation period"),
    date_range: Dict = Depends(validate_date_range),
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceRatesResponse:
    """Retrieve compliance rate trends.

    Args:
        period: Aggregation period.
        date_range: Date range filter.
        user: Authenticated user with analytics:read permission.

    Returns:
        ComplianceRatesResponse with rate trends.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_compliance_rates(
            period=period,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
        )
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return ComplianceRatesResponse(
            rates=result.get("rates", []),
            overall_compliance_rate=Decimal(str(result.get("overall_compliance_rate", 0))),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("analytics_compliance", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance rates failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance rates",
        )


@router.get(
    "/car-performance",
    response_model=CARPerformanceResponse,
    summary="CAR lifecycle analytics",
    description="Retrieve CAR lifecycle performance analytics.",
    responses={200: {"description": "CAR performance retrieved"}},
)
async def get_car_performance(
    request: Request,
    period: Optional[str] = Query(default="monthly", description="Aggregation period"),
    date_range: Dict = Depends(validate_date_range),
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> CARPerformanceResponse:
    """Retrieve CAR lifecycle performance analytics.

    Args:
        period: Aggregation period.
        date_range: Date range filter.
        user: Authenticated user with analytics:read permission.

    Returns:
        CARPerformanceResponse with lifecycle data.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_car_performance(
            period=period,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
        )
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return CARPerformanceResponse(
            performance=result.get("performance", []),
            overall_sla_compliance=Decimal(str(result.get("overall_sla_compliance", 0))),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("analytics_car_perf", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CAR performance failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve CAR performance",
        )


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Executive dashboard data",
    description="Retrieve aggregate executive dashboard data for EUDR audit management.",
    responses={200: {"description": "Dashboard data retrieved"}},
)
async def get_dashboard(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> DashboardResponse:
    """Retrieve executive dashboard aggregate data.

    Args:
        user: Authenticated user with analytics:read permission.

    Returns:
        DashboardResponse with aggregate KPIs.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_dashboard()
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return DashboardResponse(
            active_audits=result.get("active_audits", 0),
            open_cars=result.get("open_cars", 0),
            overdue_cars=result.get("overdue_cars", 0),
            car_sla_compliance_rate=Decimal(str(result.get("car_sla_compliance_rate", 0))),
            total_ncs_this_quarter=result.get("total_ncs_this_quarter", 0),
            critical_ncs_this_quarter=result.get("critical_ncs_this_quarter", 0),
            audits_completed_this_quarter=result.get("audits_completed_this_quarter", 0),
            compliance_rate=Decimal(str(result.get("compliance_rate", 0))),
            pending_authority_responses=result.get("pending_authority_responses", 0),
            active_certificates=result.get("active_certificates", 0),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("analytics_dashboard", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dashboard failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data",
        )
