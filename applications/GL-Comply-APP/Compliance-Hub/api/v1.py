# -*- coding: utf-8 -*-
"""Unified compliance FastAPI v1."""

from __future__ import annotations

import logging

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    APIRouter = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]

from agents.orchestrator_agent import ComplianceOrchestrator
from schemas.models import (
    ApplicabilityRequest,
    ApplicabilityResult,
    ComplianceRequest,
    UnifiedComplianceReport,
)
from services import applicability, registry
from services.store import JobStore, default_store

logger = logging.getLogger(__name__)


def build_router(
    orchestrator: ComplianceOrchestrator | None = None,
    store: JobStore | None = None,
):
    if APIRouter is None:
        raise RuntimeError("FastAPI not installed; install greenlang[server]")
    router = APIRouter(prefix="/compliance", tags=["compliance"])
    orch = orchestrator or ComplianceOrchestrator()
    job_store = store or default_store()

    @router.post("/intake", response_model=UnifiedComplianceReport)
    async def intake(request: ComplianceRequest) -> UnifiedComplianceReport:
        try:
            report = await orch.run(request)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        job_store.save(report)
        return report

    @router.get("/jobs/{job_id}", response_model=UnifiedComplianceReport)
    async def get_job(job_id: str) -> UnifiedComplianceReport:
        report = job_store.get(job_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return report

    @router.get("/results/{job_id}", response_model=UnifiedComplianceReport)
    async def get_result(job_id: str) -> UnifiedComplianceReport:
        return await get_job(job_id)

    @router.post("/applicability", response_model=ApplicabilityResult)
    def check_applicability(request: ApplicabilityRequest) -> ApplicabilityResult:
        return applicability.evaluate(request)

    @router.get("/frameworks")
    def list_frameworks() -> dict[str, list[str]]:
        return {"frameworks": [fw.value for fw in registry.available()]}

    return router
