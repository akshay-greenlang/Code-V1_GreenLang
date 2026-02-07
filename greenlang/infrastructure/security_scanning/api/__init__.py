# -*- coding: utf-8 -*-
"""
Security Scanning API Routes - SEC-007 Security Scanning Pipeline

FastAPI routers for security scanning endpoints:

    /api/v1/security/vulnerabilities - Vulnerability management
    /api/v1/security/scans          - Scan execution and history
    /api/v1/security/dashboard      - Dashboard and statistics

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter

    from .vulnerabilities_routes import vulnerabilities_router
    from .scans_routes import scans_router
    from .dashboard_routes import dashboard_router

    # Combined security router
    # Note: prefix is added by auth_setup.py when including the router
    security_router = APIRouter(
        tags=["Security"],
    )

    security_router.include_router(vulnerabilities_router)
    security_router.include_router(scans_router)
    security_router.include_router(dashboard_router)

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False
    security_router = None  # type: ignore[assignment]
    vulnerabilities_router = None  # type: ignore[assignment]
    scans_router = None  # type: ignore[assignment]
    dashboard_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - security routers are None")


__all__ = [
    "security_router",
    "vulnerabilities_router",
    "scans_router",
    "dashboard_router",
    "FASTAPI_AVAILABLE",
]
