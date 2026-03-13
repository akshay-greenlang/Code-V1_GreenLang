# -*- coding: utf-8 -*-
"""
API Package - AGENT-EUDR-026 Due Diligence Orchestrator

FastAPI route modules for the Due Diligence Orchestrator REST API.
All routes are prefixed with /api/v1/eudr-ddo/ when mounted by the
main application.

Route Modules:
    - workflow_routes: Workflow CRUD (create, list, get, delete, validate, clone)
    - execution_routes: Execution control (start, pause, resume, cancel, rollback)
    - status_routes: Status monitoring (status, progress, phase-status, eta)
    - quality_gate_routes: Quality gate management (list, override, details)
    - checkpoint_routes: Checkpoint management (list, create, get)
    - template_routes: Template management (list, by-commodity, create, get)
    - package_routes: Package generation and download
    - monitoring_routes: Health, metrics, version, circuit breakers, DLQ

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from greenlang.agents.eudr.due_diligence_orchestrator.api.router import (
    ddo_router,
)

__all__ = ["ddo_router"]
