# -*- coding: utf-8 -*-
"""
Supply Chain Mapper REST API - AGENT-EUDR-001

FastAPI router package providing 25+ REST endpoints for EUDR supply chain
mapping operations including graph CRUD, multi-tier mapping, traceability,
risk propagation, gap analysis, visualization, and supplier onboarding.

Route Modules:
    - graph_routes: Graph CRUD (create, list, get, delete, export)
    - mapping_routes: Multi-tier recursive mapping and tier distribution
    - traceability_routes: Forward/backward trace and batch traceability
    - risk_routes: Risk propagation, summary, and heatmap
    - gap_routes: Gap analysis, listing, and resolution
    - visualization_routes: Graph layout and Sankey diagram data
    - onboarding_routes: Supplier invitation, token access, and submission
    - router: Main router registration with /v1 prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-supply-chain:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
Status: Production Ready
"""

from greenlang.agents.eudr.supply_chain_mapper.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
