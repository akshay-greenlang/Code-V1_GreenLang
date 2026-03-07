# -*- coding: utf-8 -*-
"""
Geolocation Verification REST API - AGENT-EUDR-002

FastAPI router package providing 20+ REST endpoints for EUDR geolocation
verification operations including coordinate validation, polygon topology
verification, protected area screening, deforestation cutoff checks,
accuracy scoring, batch verification pipelines, and Article 9 compliance
reporting.

Route Modules:
    - coordinate_routes: Coordinate validation (single and batch)
    - polygon_routes: Polygon topology verification and auto-repair
    - verification_routes: Protected area, deforestation, and full plot verification
    - batch_routes: Batch verification job management
    - scoring_routes: Accuracy scoring and score weight management
    - compliance_routes: Article 9 compliance report generation
    - router: Main router registration with /v1 prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-geolocation:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
Status: Production Ready
"""

from greenlang.agents.eudr.geolocation_verification.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
