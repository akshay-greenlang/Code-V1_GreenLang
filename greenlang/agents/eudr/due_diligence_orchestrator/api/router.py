# -*- coding: utf-8 -*-
"""
Consolidated API Router - AGENT-EUDR-026 Due Diligence Orchestrator

Aggregates all sub-route modules into a single FastAPI APIRouter
mounted at /api/v1/eudr-ddo. This router is the single entry point
for registering all DDO-026 endpoints with the main FastAPI application.

Sub-Routers (8):
    - workflow_routes:      /workflows (6 endpoints)
    - execution_routes:     /workflows (5 endpoints - start/pause/resume/cancel/rollback)
    - status_routes:        /workflows (4 endpoints - status/progress/phase-status/eta)
    - quality_gate_routes:  /workflows (3 endpoints - gates/override/details)
    - checkpoint_routes:    (3 endpoints - list/create/get checkpoints)
    - template_routes:      /templates (4 endpoints)
    - package_routes:       (4 endpoints - generate/get/download/validate)
    - monitoring_routes:    (6 endpoints - health/metrics/version/circuit-breakers/dlq)

Total: 35 endpoints

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

from fastapi import APIRouter

from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
    router as workflow_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
    router as execution_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
    router as status_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
    router as quality_gate_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
    router as checkpoint_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
    router as template_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
    router as package_router,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
    router as monitoring_router,
)

# ---------------------------------------------------------------------------
# Consolidated DDO router
# ---------------------------------------------------------------------------

ddo_router = APIRouter(
    prefix="/api/v1/eudr-ddo",
    tags=["EUDR Due Diligence Orchestrator"],
)

# Include all sub-routers
ddo_router.include_router(workflow_router)
ddo_router.include_router(execution_router)
ddo_router.include_router(status_router)
ddo_router.include_router(quality_gate_router)
ddo_router.include_router(checkpoint_router)
ddo_router.include_router(template_router)
ddo_router.include_router(package_router)
ddo_router.include_router(monitoring_router)

__all__ = ["ddo_router"]
