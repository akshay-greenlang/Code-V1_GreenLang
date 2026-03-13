# -*- coding: utf-8 -*-
"""
Segregation Verifier REST API - AGENT-EUDR-010

FastAPI router package providing 37 REST endpoints for EUDR segregation
verification operations including segregation control points (SCPs),
storage zone management, transport segregation, processing line
verification, contamination detection, facility assessment, reporting,
batch jobs, and health.

Route Modules:
    - scp_routes: SCP registration, update, validation, search, batch import
    - storage_routes: Storage zone registration, events, audits, scoring
    - transport_routes: Vehicle registration, verification, cleaning, history
    - processing_routes: Processing line registration, changeover, verification
    - contamination_routes: Detection, event recording, impact assessment
    - assessment_routes: Facility assessment, history
    - report_routes: Audit, contamination, and evidence reports
    - router: Main router with batch jobs and health endpoints

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-sgv:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Segregation Verifier)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from greenlang.agents.eudr.segregation_verifier.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
