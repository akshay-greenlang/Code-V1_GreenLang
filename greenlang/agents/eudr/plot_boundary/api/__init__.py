# -*- coding: utf-8 -*-
"""
Plot Boundary Manager REST API - AGENT-EUDR-006

FastAPI router package providing 30 REST endpoints for EUDR plot boundary
management operations including boundary CRUD, polygon topology validation,
geodetic area calculation, overlap detection, immutable version history,
multi-format export, and split/merge with genealogy tracking.

Route Modules:
    - boundary_routes: Boundary CRUD (create, get, update, delete, batch, search)
    - validation_routes: Polygon topology validation and auto-repair
    - area_routes: Geodetic area calculation and EUDR 4ha threshold checks
    - overlap_routes: Spatial overlap detection, scanning, and resolution
    - version_routes: Immutable version history, date queries, diff, lineage
    - export_routes: Multi-format export, split, merge, and genealogy
    - router: Main router registration with /v1/eudr-pbm prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-boundary:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from greenlang.agents.eudr.plot_boundary.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
