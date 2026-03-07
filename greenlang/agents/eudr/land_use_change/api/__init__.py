# -*- coding: utf-8 -*-
"""
Land Use Change Detector REST API - AGENT-EUDR-005

FastAPI router package providing 33 REST endpoints for EUDR land use
change detection operations including multi-class classification,
temporal transition detection, trajectory analysis, EUDR cutoff date
compliance verification, conversion risk assessment, urban encroachment
analysis, compliance report generation, and batch processing.

Route Modules:
    - classification_routes: Land use classification, batch, history, compare
    - transition_routes: Transition detection, batch, matrix, types
    - trajectory_routes: Trajectory analysis, batch, retrieval
    - verification_routes: Cutoff verification, batch, evidence, complete
    - risk_routes: Risk assessment, urban analysis, batch jobs
    - report_routes: Report generation, retrieval, download, batch
    - router: Main router registration with /v1/eudr-luc prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-luc:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
Status: Production Ready
"""

from greenlang.agents.eudr.land_use_change.api.router import (
    router,
    get_eudr_luc_router,
)

__all__ = [
    "router",
    "get_eudr_luc_router",
]
