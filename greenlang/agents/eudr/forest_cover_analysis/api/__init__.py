# -*- coding: utf-8 -*-
"""
Forest Cover Analysis REST API - AGENT-EUDR-004

FastAPI router package providing 30 REST endpoints for EUDR forest cover
analysis operations including canopy density estimation, forest type
classification, historical cover reconstruction, deforestation-free
verification, canopy height estimation, fragmentation analysis, biomass
estimation, compliance report generation, and batch analysis.

Route Modules:
    - density_routes: Canopy density analysis, batch, history, comparison
    - classification_routes: Forest type classification, batch, types list
    - historical_routes: Historical cover reconstruction, batch, sources
    - verification_routes: Deforestation-free verification, evidence, complete
    - analysis_routes: Height, fragmentation, biomass, profile, comparison
    - report_routes: Compliance report generation, retrieval, download, batch
    - router: Main router registration with /v1/eudr-fca prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-fca:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
Status: Production Ready
"""

from greenlang.agents.eudr.forest_cover_analysis.api.router import (
    router,
    get_eudr_fca_router,
)

__all__ = [
    "router",
    "get_eudr_fca_router",
]
