# -*- coding: utf-8 -*-
"""
Dual Reporting Reconciliation REST API Router - AGENT-MRV-013
===============================================================

16 REST endpoints for the Dual Reporting Reconciliation Agent
(GL-MRV-X-024).

Prefix: ``/api/v1/dual-reporting``

Endpoints:
     1. POST   /reconciliations                - Execute single reconciliation
     2. POST   /reconciliations/batch          - Execute batch reconciliation
     3. GET    /reconciliations                - List reconciliations with filters
     4. GET    /reconciliations/{id}           - Get reconciliation by ID
     5. DELETE /reconciliations/{id}           - Delete reconciliation
     6. GET    /reconciliations/{id}/discrepancies - List discrepancies
     7. GET    /reconciliations/{id}/waterfall     - Get waterfall decomposition
     8. GET    /reconciliations/{id}/quality       - Get quality assessment
     9. GET    /reconciliations/{id}/tables        - Get reporting tables
    10. GET    /reconciliations/{id}/trends        - Get trend analysis
    11. POST   /reconciliations/{id}/compliance    - Run compliance check
    12. GET    /compliance/{id}                - Get compliance result
    13. GET    /aggregations                   - Get aggregated emissions
    14. POST   /export                         - Export reconciliation report
    15. GET    /health                         - Service health check
    16. GET    /stats                          - Service statistics

GHG Protocol Scope 2 Guidance dual-reporting reconciliation between
location-based and market-based emission accounting methods.

All emission values use Python ``Decimal`` for zero-hallucination
deterministic arithmetic. No LLM involvement in any calculation path.
Every result carries a SHA-256 provenance hash for full audit trails.

Supported Regulatory Frameworks (7):
    - GHG Protocol Scope 2 Guidance (2015)
    - CSRD/ESRS E1
    - CDP Climate Change
    - SBTi
    - GRI 305
    - ISO 14064-1:2018
    - RE100

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI/Pydantic imports
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; dual-reporting router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    # ---------------------------------------------------------------
    # Reconciliation request models
    # ---------------------------------------------------------------

    class SingleReconciliationRequest(BaseModel):
        """Request body for a single dual-reporting reconciliation.

        Takes upstream location-based and market-based emission results
        and reconciles the two totals, identifying discrepancies, scoring
        quality, and generating framework-specific reporting tables.
        """

        tenant_id: str = Field(
            ...,
            min_length=1,
            description="Owning tenant identifier for multi-tenancy",
        )
        period_start: str = Field(
            ...,
            description="Start date of the reporting period (ISO-8601)",
        )
        period_end: str = Field(
            ...,
            description="End date of the reporting period (ISO-8601)",
        )
        upstream_results: List[Dict[str, Any]] = Field(
            ...,
            min_length=1,
            description=(
                "List of upstream emission results from Scope 2 agents "
                "(MRV-009 through MRV-012). Each dict must include "
                "facility_id, energy_type, method, emissions_tco2e, "
                "and ef_source"
            ),
        )
        frameworks: Optional[List[str]] = Field(
            default=None,
            description=(
                "Regulatory frameworks to check. Empty means all 7. "
                "Options: ghg_protocol, csrd_esrs, cdp, sbti, gri, "
                "iso_14064, re100"
            ),
        )
        trend_data: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description=(
                "Historical trend data points for multi-period analysis. "
                "Each dict should include period, location_tco2e, "
                "market_tco2e"
            ),
        )
        include_trends: Optional[bool] = Field(
            default=True,
            description="Whether to include trend analysis in results",
        )
        include_compliance: Optional[bool] = Field(
            default=True,
            description="Whether to include compliance checking",
        )
        reconciliation_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional reconciliation identifier "
                "(auto-generated if omitted)"
            ),
        )
        revenue: Optional[float] = Field(
            default=None,
            ge=0,
            description="Revenue in reporting currency for intensity metrics",
        )
        fte_count: Optional[int] = Field(
            default=None,
            ge=0,
            description="Full-time-equivalent headcount for intensity metrics",
        )
        floor_area_sqm: Optional[float] = Field(
            default=None,
            ge=0,
            description="Floor area in square metres for intensity metrics",
        )
        production_units: Optional[float] = Field(
            default=None,
            ge=0,
            description="Production unit count for intensity metrics",
        )

    class BatchReconciliationBody(BaseModel):
        """Request body for batch dual-reporting reconciliation.

        Processes multiple reporting periods in a single batch. Each
        period follows the same schema as the single reconciliation
        request.
        """

        batch_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional batch identifier (auto-generated if omitted)"
            ),
        )
        tenant_id: str = Field(
            ...,
            min_length=1,
            description="Owning tenant identifier for multi-tenancy",
        )
        periods: List[Dict[str, Any]] = Field(
            ...,
            min_length=1,
            description=(
                "List of period request dictionaries, each containing "
                "period_start, period_end, upstream_results, and "
                "optional fields"
            ),
        )

    # ---------------------------------------------------------------
    # Compliance request model
    # ---------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a regulatory compliance check.

        Evaluates a completed reconciliation against one or more
        regulatory frameworks.
        """

        frameworks: List[str] = Field(
            default_factory=list,
            description=(
                "Regulatory frameworks to check. Empty means all 7. "
                "Options: ghg_protocol, csrd_esrs, cdp, sbti, gri, "
                "iso_14064, re100"
            ),
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    # ---------------------------------------------------------------
    # Export request model
    # ---------------------------------------------------------------

    class ExportBody(BaseModel):
        """Request body for exporting a reconciliation report."""

        reconciliation_id: str = Field(
            ...,
            min_length=1,
            description="ID of the reconciliation to export",
        )
        format: str = Field(
            default="json",
            description="Export format: json, csv, xlsx, or pdf",
        )
        frameworks: Optional[List[str]] = Field(
            default=None,
            description="Optional filter for specific framework tables",
        )


# ===================================================================
# Serialisation helper
# ===================================================================


def _serialize_result(data: Any) -> Dict[str, Any]:
    """Serialize a result to a JSON-safe dictionary.

    Handles Pydantic models, Decimal values, and nested structures.

    Args:
        data: Result data to serialize.

    Returns:
        JSON-safe dictionary.
    """
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")
    if isinstance(data, dict):
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, Decimal):
                result[k] = float(v)
            elif hasattr(v, "model_dump"):
                result[k] = v.model_dump(mode="json")
            elif isinstance(v, list):
                result[k] = [
                    _serialize_result(item) if isinstance(item, (dict,))
                    else (float(item) if isinstance(item, Decimal) else item)
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = _serialize_result(v)
            else:
                result[k] = v
        return result
    return data


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Dual Reporting Reconciliation FastAPI APIRouter.

    Returns:
        Configured APIRouter with 16 endpoints covering reconciliations,
        discrepancies, quality, tables, trends, compliance, aggregations,
        exports, health, and statistics.

    Raises:
        RuntimeError: If FastAPI is not installed in the environment.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the dual-reporting router"
        )

    router = APIRouter(
        prefix="/api/v1/dual-reporting",
        tags=["Dual Reporting Reconciliation"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the DualReportingService singleton.

        Returns the initialized service. Raises HTTPException 503
        if the service has not been initialized.
        """
        try:
            from greenlang.dual_reporting_reconciliation.setup import (
                get_service,
            )
            svc = get_service()
            if svc is not None:
                return svc
        except (ImportError, AttributeError):
            pass

        # Fallback: try pipeline engine directly
        try:
            from greenlang.dual_reporting_reconciliation.dual_reporting_pipeline import (
                DualReportingPipelineEngine,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Dual reporting service not available",
            )

        if not hasattr(_get_service, "_instance"):
            _get_service._instance = DualReportingPipelineEngine()
        return _get_service._instance

    # ==================================================================
    # 1. POST /reconciliations - Execute single reconciliation
    # ==================================================================

    @router.post("/reconciliations", status_code=201)
    async def create_reconciliation(
        body: SingleReconciliationRequest,
    ) -> Dict[str, Any]:
        """Execute a single dual-reporting reconciliation.

        Reconciles location-based and market-based Scope 2 emission
        totals, identifying discrepancies, scoring data quality,
        generating framework-specific reporting tables, analysing
        trends, and checking regulatory compliance.

        Permission: dual-reporting:reconcile
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "tenant_id": body.tenant_id,
            "period_start": body.period_start,
            "period_end": body.period_end,
            "upstream_results": body.upstream_results,
            "include_trends": body.include_trends or True,
            "include_compliance": body.include_compliance or True,
        }

        if body.reconciliation_id is not None:
            request_data["reconciliation_id"] = body.reconciliation_id
        if body.frameworks is not None:
            request_data["frameworks"] = body.frameworks
        if body.trend_data is not None:
            request_data["trend_data"] = body.trend_data
        if body.revenue is not None:
            request_data["revenue"] = body.revenue
        if body.fte_count is not None:
            request_data["fte_count"] = body.fte_count
        if body.floor_area_sqm is not None:
            request_data["floor_area_sqm"] = body.floor_area_sqm
        if body.production_units is not None:
            request_data["production_units"] = body.production_units

        try:
            result = svc.reconcile(request_data)
            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_reconciliation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /reconciliations/batch - Execute batch reconciliation
    # ==================================================================

    @router.post("/reconciliations/batch", status_code=201)
    async def create_batch_reconciliation(
        body: BatchReconciliationBody,
    ) -> Dict[str, Any]:
        """Execute batch dual-reporting reconciliation.

        Processes multiple reporting periods in a single batch.

        Permission: dual-reporting:reconcile
        """
        svc = _get_service()

        batch_data: Dict[str, Any] = {
            "batch_id": body.batch_id or str(uuid.uuid4()),
            "tenant_id": body.tenant_id,
            "periods": body.periods,
        }

        try:
            result = svc.reconcile_batch(batch_data)
            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_batch_reconciliation failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /reconciliations - List reconciliations with filters
    # ==================================================================

    @router.get("/reconciliations", status_code=200)
    async def list_reconciliations(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        skip: int = Query(
            0, ge=0,
            description="Number of records to skip (offset)",
        ),
        limit: int = Query(
            20, ge=1, le=100,
            description="Maximum number of records to return",
        ),
    ) -> Dict[str, Any]:
        """List reconciliation results with pagination.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.list_reconciliations(
                tenant_id=tenant_id,
                skip=skip,
                limit=limit,
            )
            return _serialize_result(result)

        except Exception as exc:
            logger.error(
                "list_reconciliations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. GET /reconciliations/{id} - Get reconciliation by ID
    # ==================================================================

    @router.get("/reconciliations/{recon_id}", status_code=200)
    async def get_reconciliation(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a reconciliation result by its unique identifier.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        result = svc.get_reconciliation(recon_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Reconciliation not found: {recon_id}",
            )
        return _serialize_result(result)

    # ==================================================================
    # 5. DELETE /reconciliations/{id} - Delete reconciliation
    # ==================================================================

    @router.delete("/reconciliations/{recon_id}", status_code=200)
    async def delete_reconciliation(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """Delete a reconciliation result.

        Permission: dual-reporting:delete
        """
        svc = _get_service()

        deleted = svc.delete_reconciliation(recon_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Reconciliation not found: {recon_id}",
            )
        return {"success": True, "reconciliation_id": recon_id}

    # ==================================================================
    # 6. GET /reconciliations/{id}/discrepancies
    # ==================================================================

    @router.get(
        "/reconciliations/{recon_id}/discrepancies",
        status_code=200,
    )
    async def list_discrepancies(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """List discrepancies found in a reconciliation.

        Returns all identified discrepancies between location-based
        and market-based totals with root cause classification.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.list_discrepancies(recon_id)
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "list_discrepancies failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /reconciliations/{id}/waterfall
    # ==================================================================

    @router.get(
        "/reconciliations/{recon_id}/waterfall",
        status_code=200,
    )
    async def get_waterfall(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get waterfall decomposition for a reconciliation.

        Returns a bridge chart from location-based total to market-based
        total showing each discrepancy driver contribution.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.get_waterfall(recon_id)
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "get_waterfall failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. GET /reconciliations/{id}/quality
    # ==================================================================

    @router.get(
        "/reconciliations/{recon_id}/quality",
        status_code=200,
    )
    async def get_quality_assessment(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get quality assessment for a reconciliation.

        Returns composite quality score, dimensional breakdowns, and
        emission factor hierarchy quality scores.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.get_quality_assessment(recon_id)
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "get_quality_assessment failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /reconciliations/{id}/tables
    # ==================================================================

    @router.get(
        "/reconciliations/{recon_id}/tables",
        status_code=200,
    )
    async def get_reporting_tables(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
        frameworks: Optional[str] = Query(
            None,
            description=(
                "Comma-separated framework filter (e.g. "
                "'ghg_protocol,csrd_esrs')"
            ),
        ),
    ) -> Dict[str, Any]:
        """Get multi-framework reporting tables for a reconciliation.

        Returns formatted reporting tables for GHG Protocol, CSRD/ESRS,
        CDP, SBTi, GRI, ISO 14064, and RE100.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        fw_list = None
        if frameworks:
            fw_list = [
                f.strip() for f in frameworks.split(",")
                if f.strip()
            ]

        try:
            result = svc.get_reporting_tables(recon_id, frameworks=fw_list)
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "get_reporting_tables failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /reconciliations/{id}/trends
    # ==================================================================

    @router.get(
        "/reconciliations/{recon_id}/trends",
        status_code=200,
    )
    async def get_trend_analysis(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get trend analysis for a reconciliation.

        Returns YoY, CAGR, PIF, RE100, SBTi tracking, and intensity
        metrics for multi-period trend analysis.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.get_trend_analysis(recon_id)
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "get_trend_analysis failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. POST /reconciliations/{id}/compliance
    # ==================================================================

    @router.post(
        "/reconciliations/{recon_id}/compliance",
        status_code=201,
    )
    async def check_compliance(
        recon_id: str = Path(
            ..., description="Reconciliation identifier",
        ),
        body: ComplianceCheckBody = None,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check for a reconciliation.

        Evaluates the reconciliation results against the specified
        regulatory frameworks (or all 7 if none specified).

        Permission: dual-reporting:compliance
        """
        svc = _get_service()

        frameworks = (
            body.frameworks if body and body.frameworks else None
        )

        try:
            result = svc.check_compliance(
                recon_id, frameworks=frameworks,
            )
            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /compliance/{id}
    # ==================================================================

    @router.get("/compliance/{compliance_id}", status_code=200)
    async def get_compliance_result(
        compliance_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a compliance check result by its identifier.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        result = svc.get_compliance_result(compliance_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Compliance result not found: {compliance_id}",
            )
        return _serialize_result(result)

    # ==================================================================
    # 13. GET /aggregations
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        group_by: str = Query(
            "energy_type",
            description=(
                "Dimension to group by: energy_type, facility, "
                "region, business_unit, period"
            ),
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated reconciliation data.

        Returns portfolio-level aggregations grouped by the specified
        dimension.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.get_aggregations(
                group_by=group_by, tenant_id=tenant_id,
            )
            return _serialize_result(result)

        except Exception as exc:
            logger.error(
                "get_aggregations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. POST /export
    # ==================================================================

    @router.post("/export", status_code=201)
    async def export_report(
        body: ExportBody,
    ) -> Dict[str, Any]:
        """Export a reconciliation report.

        Generates a formatted export of reconciliation results in JSON,
        CSV, XLSX, or PDF format.

        Permission: dual-reporting:export
        """
        svc = _get_service()

        try:
            result = svc.export_report(
                reconciliation_id=body.reconciliation_id,
                export_format=body.format,
            )
            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "export_report failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. GET /health
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Service health check.

        Returns engine availability status and uptime information.
        No authentication required.
        """
        svc = _get_service()

        try:
            result = svc.health_check()
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "health_check failed: %s", exc, exc_info=True,
            )
            return {
                "status": "unhealthy",
                "error": str(exc),
            }

    # ==================================================================
    # 16. GET /stats
    # ==================================================================

    @router.get("/stats", status_code=200)
    async def get_stats() -> Dict[str, Any]:
        """Service aggregate statistics.

        Returns cumulative counters, average quality scores, and
        portfolio PIF.

        Permission: dual-reporting:read
        """
        svc = _get_service()

        try:
            result = svc.get_stats()
            return _serialize_result(result)
        except Exception as exc:
            logger.error(
                "get_stats failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    return router


# ===================================================================
# Module-level router instance
# ===================================================================

try:
    router = create_router()
except RuntimeError:
    router = None
    logger.debug(
        "FastAPI not available; dual-reporting router not created"
    )


__all__ = [
    "router",
    "create_router",
]
