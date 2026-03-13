# -*- coding: utf-8 -*-
"""
AGENT-EUDR-022: Protected Area Validator API Module

This package provides the FastAPI router and endpoints for the Protected Area
Validator agent, which validates plots against protected areas (WDPA, national
parks, UNESCO sites, etc.) and monitors buffer zones, PADDD events, and violations.

Route Structure:
    /v1/eudr-pav/protected-areas     - Protected area database management
    /v1/eudr-pav/overlap              - Spatial overlap detection and analysis
    /v1/eudr-pav/buffer-zones         - Buffer zone monitoring
    /v1/eudr-pav/designation          - Designation status validation
    /v1/eudr-pav/risk                 - Risk scoring and proximity alerts
    /v1/eudr-pav/violations           - Violation detection and management
    /v1/eudr-pav/compliance           - Compliance assessment and reporting
    /v1/eudr-pav/paddd                - PADDD event monitoring
    /v1/eudr-pav/health               - Health check endpoint

Auth & RBAC:
    All endpoints require JWT authentication (SEC-001) and appropriate
    eudr-pav:* permissions (SEC-002 RBAC).

Usage:
    >>> from greenlang.agents.eudr.protected_area_validator.api import get_router
    >>> app.include_router(get_router(), prefix="/api")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
"""

from __future__ import annotations

from greenlang.agents.eudr.protected_area_validator.api.router import (
    get_router,
    router,
)

__all__ = [
    "get_router",
    "router",
]
