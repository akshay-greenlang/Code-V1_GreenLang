# -*- coding: utf-8 -*-
"""Scope Engine FastAPI router.

Endpoints:
- POST /scope-engine/compute                     -> ComputationResponse
- GET  /scope-engine/results/{id}                -> ComputationResponse
- GET  /scope-engine/frameworks                  -> list of available frameworks
- GET  /scope-engine/frameworks/{fw}/view/{id}   -> FrameworkView
"""

from __future__ import annotations

import logging

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    APIRouter = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]

from greenlang.scope_engine import adapters as framework_adapters
from greenlang.scope_engine.models import (
    ComputationRequest,
    ComputationResponse,
    Framework,
    FrameworkView,
)
from greenlang.scope_engine.provenance import ProvenanceRecorder
from greenlang.scope_engine.service import ScopeEngineService
from greenlang.scope_engine.store import InMemoryStore, get_default_store

logger = logging.getLogger(__name__)


def build_router(
    service: ScopeEngineService | None = None,
    store: InMemoryStore | None = None,
    provenance: ProvenanceRecorder | None = None,
):
    if APIRouter is None:
        raise RuntimeError(
            "FastAPI not installed; install greenlang with extras [server]"
        )
    router = APIRouter(prefix="/scope-engine", tags=["scope-engine"])
    svc = service or ScopeEngineService()
    _store = store or get_default_store()
    _prov = provenance or ProvenanceRecorder()

    @router.post("/compute", response_model=ComputationResponse)
    def compute(request: ComputationRequest) -> ComputationResponse:
        try:
            response = svc.compute(request)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        _prov.record_computation(response.computation)
        _store.put(response)
        return response

    @router.get("/results/{computation_id}", response_model=ComputationResponse)
    def get_result(computation_id: str) -> ComputationResponse:
        response = _store.get(computation_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Unknown computation_id")
        return response

    @router.get("/frameworks")
    def list_frameworks() -> dict[str, list[str]]:
        return {"frameworks": [fw.value for fw in framework_adapters.available()]}

    @router.get(
        "/frameworks/{framework}/view/{computation_id}",
        response_model=FrameworkView,
    )
    def framework_view(framework: Framework, computation_id: str) -> FrameworkView:
        response = _store.get(computation_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Unknown computation_id")
        if framework in response.framework_views:
            return response.framework_views[framework]
        try:
            adapter = framework_adapters.get(framework)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return adapter.project(response.computation)

    return router
