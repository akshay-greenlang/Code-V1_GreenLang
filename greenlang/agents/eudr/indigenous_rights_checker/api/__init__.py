# -*- coding: utf-8 -*-
"""
AGENT-EUDR-021: Indigenous Rights Checker - API Package

FastAPI router package providing 25+ REST endpoints for EUDR indigenous
rights compliance operations including territory management, FPIC (Free,
Prior and Informed Consent) verification, land rights overlap analysis,
community consultation tracking, rights violation detection and resolution,
indigenous community registry, and compliance reporting.

Route Modules:
    - territory_routes: Territory management (register, list, detail, update, archive)
    - fpic_routes: FPIC verification (verify, documents, document detail, score)
    - overlap_routes: Land rights overlap (analyze, by-plot, by-territory, bulk)
    - consultation_routes: Community consultations (record, list, detail)
    - violation_routes: Rights violations (detect, list, detail, resolve)
    - registry_routes: Indigenous community registry (register, list, detail)
    - compliance_routes: Compliance reporting (report, assess)
    - router: Main router registration with /v1/eudr-irc prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with 14 eudr-irc:* permissions
    - Rate limiting via middleware decorator (100/30/10/5 req/min tiers)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
Status: Production Ready
"""

from greenlang.agents.eudr.indigenous_rights_checker.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
