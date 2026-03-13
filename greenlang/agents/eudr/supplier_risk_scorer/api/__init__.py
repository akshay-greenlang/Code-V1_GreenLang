# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer API Package - AGENT-EUDR-017

FastAPI REST API for the EUDR Supplier Risk Scorer Agent providing
composite supplier risk scoring, due diligence tracking, documentation
analysis, certification validation, geographic sourcing analysis,
supplier network analysis, continuous monitoring, and risk reporting.

Public API:
    - get_router(): Returns configured APIRouter for mounting in FastAPI app
    - router: Pre-configured APIRouter instance

Modules:
    - router: Main router aggregating 8 sub-routers
    - schemas: Pydantic v2 request/response models
    - dependencies: FastAPI dependency injection providers
    - supplier_routes: Supplier risk assessment endpoints
    - due_diligence_routes: Due diligence tracking endpoints
    - documentation_routes: Documentation analysis endpoints
    - certification_routes: Certification validation endpoints
    - geographic_routes: Geographic sourcing endpoints
    - network_routes: Supplier network endpoints
    - monitoring_routes: Continuous monitoring endpoints
    - report_routes: Risk reporting endpoints

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.agents.eudr.supplier_risk_scorer.api.router import (
    get_router,
    router,
)

__version__ = "1.0.0"

__all__ = [
    "get_router",
    "router",
    "__version__",
]
