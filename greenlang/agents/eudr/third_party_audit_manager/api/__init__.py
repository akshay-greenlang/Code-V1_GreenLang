# -*- coding: utf-8 -*-
"""
AGENT-EUDR-024: Third-Party Audit Manager - API Package

FastAPI router package providing ~39 REST endpoints for EUDR third-party
audit lifecycle management including audit planning, auditor registry,
checklist management, evidence collection, non-conformance detection,
corrective action request (CAR) management, certification scheme
integration, ISO 19011 report generation, competent authority liaison,
and audit analytics.

Route Modules (9 domain modules + main router):
    - audit_routes: Audit CRUD + scheduling (6 endpoints)
    - auditor_routes: Auditor registry + matching (5 endpoints)
    - checklist_routes: Checklist management (3 endpoints)
    - evidence_routes: Evidence collection (3 endpoints)
    - nc_routes: Non-conformance management (5 endpoints)
    - car_routes: CAR lifecycle (6 endpoints)
    - certificate_routes: Certification integration (4 endpoints)
    - report_routes: Report generation (4 endpoints)
    - authority_routes: Competent authority + analytics (5 endpoints)
    - router: Main router with /v1/eudr-tam prefix + health endpoint

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-tam:* permissions (22 permissions)
    - Rate limiting via in-memory sliding-window limiter

API Prefix: /v1/eudr-tam
RBAC Prefix: eudr-tam:
Total Endpoints: ~41 (39 domain + health + stats)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
Status: Production Ready
"""

from greenlang.agents.eudr.third_party_audit_manager.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
