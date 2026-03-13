# -*- coding: utf-8 -*-
"""
Multi-Tier Supplier Tracker REST API - AGENT-EUDR-008

FastAPI router package providing 33 REST endpoints for EUDR multi-tier
supplier tracking operations including supplier discovery, profile
management, tier depth tracking, relationship lifecycle, risk
propagation, compliance monitoring, gap analysis, audit reporting,
and batch processing.

Route Modules:
    - discovery_routes: Supplier discovery (single, batch, declaration, questionnaire)
    - profile_routes: Profile CRUD (create, get, update, deactivate, search, batch)
    - tier_routes: Tier depth and relationship management
    - compliance_routes: Risk assessment/propagation and compliance monitoring
    - report_routes: Audit reports, tier summary, gap analysis, download
    - batch_routes: Batch job submission, status, cancellation
    - router: Main router registration with /v1/eudr-mst prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-mst:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from greenlang.agents.eudr.multi_tier_supplier.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
