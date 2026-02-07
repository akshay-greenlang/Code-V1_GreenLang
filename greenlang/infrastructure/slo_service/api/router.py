# -*- coding: utf-8 -*-
"""
SLO Service REST API Router - OBS-005: SLO/SLI Definitions & Error Budget Management

FastAPI router providing CRUD operations on SLOs, error budget queries,
burn rate lookups, compliance reports, recording/alert rule generation,
dashboard generation, and health checks.

All endpoints are mounted under ``/api/v1/slos``.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; slo_router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateSLORequest(BaseModel):
        """Request body for creating a new SLO."""
        slo_id: str = Field(..., description="Unique SLO identifier")
        name: str = Field(..., description="Human-readable SLO name")
        service: str = Field(..., description="Service name")
        target: float = Field(99.9, ge=0.0, le=100.0)
        window: str = Field("30d", description="SLO window")
        description: str = Field("")
        team: str = Field("")
        labels: Dict[str, str] = Field(default_factory=dict)
        sli: Dict[str, Any] = Field(..., description="SLI definition")

    class UpdateSLORequest(BaseModel):
        """Request body for updating an SLO."""
        name: Optional[str] = None
        target: Optional[float] = Field(None, ge=0.0, le=100.0)
        description: Optional[str] = None
        team: Optional[str] = None
        enabled: Optional[bool] = None

    class ImportSLOsRequest(BaseModel):
        """Request body for importing SLOs from YAML."""
        yaml_path: str = Field(..., description="Path to YAML file")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_service(request: "Request") -> Any:
    """Extract the SLO service from app state."""
    svc = getattr(request.app.state, "slo_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="SLO service not configured",
        )
    return svc


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    slo_router = APIRouter(
        prefix="/api/v1/slos",
        tags=["slos"],
    )

    @slo_router.get("", summary="List SLOs")
    async def list_slos(
        request: Request,
        service: Optional[str] = Query(None),
        team: Optional[str] = Query(None),
    ) -> JSONResponse:
        """List all SLO definitions with optional filters."""
        svc = _get_service(request)
        slos = svc.manager.list_all(service=service, team=team)
        return JSONResponse(
            content={
                "slos": [s.to_dict() for s in slos],
                "count": len(slos),
            }
        )

    @slo_router.post("", status_code=201, summary="Create SLO")
    async def create_slo(
        body: CreateSLORequest,
        request: Request,
    ) -> JSONResponse:
        """Create a new SLO definition."""
        from greenlang.infrastructure.slo_service.models import SLO, SLI

        svc = _get_service(request)
        sli = SLI.from_dict(body.sli)
        slo = SLO(
            slo_id=body.slo_id,
            name=body.name,
            service=body.service,
            sli=sli,
            target=body.target,
            description=body.description,
            team=body.team,
            labels=body.labels,
        )
        try:
            created = svc.manager.create(slo)
            return JSONResponse(status_code=201, content=created.to_dict())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @slo_router.get("/overview", summary="SLO overview")
    async def slo_overview(request: Request) -> JSONResponse:
        """Get a high-level overview of all SLOs."""
        svc = _get_service(request)
        slos = svc.manager.list_all()
        return JSONResponse(
            content={
                "total_slos": len(slos),
                "enabled": sum(1 for s in slos if s.enabled),
                "disabled": sum(1 for s in slos if not s.enabled),
            }
        )

    @slo_router.get("/budgets", summary="All budgets")
    async def all_budgets(request: Request) -> JSONResponse:
        """Get error budgets for all SLOs."""
        svc = _get_service(request)
        budgets = svc.get_all_budgets()
        return JSONResponse(
            content={
                "budgets": [b.to_dict() for b in budgets],
                "count": len(budgets),
            }
        )

    @slo_router.get("/compliance", summary="Compliance report")
    async def compliance_report(
        request: Request,
        report_type: str = Query("weekly"),
    ) -> JSONResponse:
        """Generate a compliance report."""
        svc = _get_service(request)
        report = svc.generate_compliance_report(report_type)
        return JSONResponse(content=report.to_dict())

    @slo_router.post("/recording-rules", summary="Generate recording rules")
    async def generate_recording_rules(request: Request) -> JSONResponse:
        """Generate and write Prometheus recording rules."""
        svc = _get_service(request)
        path = svc.generate_recording_rules()
        return JSONResponse(content={"path": path, "status": "generated"})

    @slo_router.post("/alert-rules", summary="Generate alert rules")
    async def generate_alert_rules(request: Request) -> JSONResponse:
        """Generate and write Prometheus alert rules."""
        svc = _get_service(request)
        path = svc.generate_alert_rules()
        return JSONResponse(content={"path": path, "status": "generated"})

    @slo_router.post("/dashboards", summary="Generate dashboards")
    async def generate_dashboards(request: Request) -> JSONResponse:
        """Generate and write Grafana dashboards."""
        svc = _get_service(request)
        paths = svc.generate_dashboards()
        return JSONResponse(content={"paths": paths, "status": "generated"})

    @slo_router.post("/evaluate", summary="Evaluate SLOs")
    async def evaluate_slos(request: Request) -> JSONResponse:
        """Trigger an SLO evaluation cycle."""
        svc = _get_service(request)
        results = await svc.evaluate_all()
        return JSONResponse(content={"evaluated": len(results), "results": results})

    @slo_router.post("/import", summary="Import SLOs from YAML")
    async def import_slos(
        body: ImportSLOsRequest,
        request: Request,
    ) -> JSONResponse:
        """Import SLO definitions from a YAML file."""
        svc = _get_service(request)
        try:
            loaded = svc.manager.load_from_yaml(body.yaml_path)
            return JSONResponse(
                content={"imported": len(loaded), "slos": [s.to_dict() for s in loaded]}
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="YAML file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @slo_router.get("/export", summary="Export SLOs to YAML")
    async def export_slos(request: Request) -> JSONResponse:
        """Export all SLOs as a YAML-compatible structure."""
        svc = _get_service(request)
        slos = svc.manager.list_all()
        return JSONResponse(
            content={"slos": [s.to_dict() for s in slos]}
        )

    @slo_router.get("/health", summary="Health check")
    async def health_check(request: Request) -> JSONResponse:
        """Service health check."""
        return JSONResponse(
            content={"status": "healthy", "service": "slo-service"}
        )

    @slo_router.get("/{slo_id}", summary="Get SLO by ID")
    async def get_slo(slo_id: str, request: Request) -> JSONResponse:
        """Get a single SLO by ID."""
        svc = _get_service(request)
        slo = svc.manager.get(slo_id)
        if slo is None:
            raise HTTPException(status_code=404, detail="SLO not found")
        return JSONResponse(content=slo.to_dict())

    @slo_router.patch("/{slo_id}", summary="Update SLO")
    async def update_slo(
        slo_id: str,
        body: UpdateSLORequest,
        request: Request,
    ) -> JSONResponse:
        """Update an SLO definition."""
        svc = _get_service(request)
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        try:
            updated = svc.manager.update(slo_id, updates)
            return JSONResponse(content=updated.to_dict())
        except KeyError:
            raise HTTPException(status_code=404, detail="SLO not found")

    @slo_router.delete("/{slo_id}", summary="Delete SLO")
    async def delete_slo(slo_id: str, request: Request) -> JSONResponse:
        """Soft-delete an SLO."""
        svc = _get_service(request)
        deleted = svc.manager.delete(slo_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="SLO not found")
        return JSONResponse(content={"deleted": True, "slo_id": slo_id})

    @slo_router.get("/{slo_id}/history", summary="SLO version history")
    async def get_slo_history(slo_id: str, request: Request) -> JSONResponse:
        """Get version history for an SLO."""
        svc = _get_service(request)
        history = svc.manager.get_history(slo_id)
        return JSONResponse(content={"slo_id": slo_id, "history": history})

    @slo_router.get("/{slo_id}/budget", summary="Get error budget")
    async def get_error_budget(slo_id: str, request: Request) -> JSONResponse:
        """Get the current error budget for an SLO."""
        svc = _get_service(request)
        budget = svc.get_budget(slo_id)
        if budget is None:
            raise HTTPException(status_code=404, detail="Budget not found")
        return JSONResponse(content=budget.to_dict())

    @slo_router.get("/{slo_id}/budget/history", summary="Budget history")
    async def get_budget_history(
        slo_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get error budget history for an SLO."""
        svc = _get_service(request)
        history = svc.get_budget_history(slo_id)
        return JSONResponse(
            content={"slo_id": slo_id, "history": [b.to_dict() for b in history]}
        )

    @slo_router.get("/{slo_id}/burn-rate", summary="Get burn rate")
    async def get_burn_rate(slo_id: str, request: Request) -> JSONResponse:
        """Get current burn rates for an SLO."""
        svc = _get_service(request)
        rates = svc.get_burn_rates(slo_id)
        if rates is None:
            raise HTTPException(status_code=404, detail="SLO not found")
        return JSONResponse(content={"slo_id": slo_id, "burn_rates": rates})

else:
    slo_router = None  # type: ignore[assignment]
