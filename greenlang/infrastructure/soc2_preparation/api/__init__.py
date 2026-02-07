# -*- coding: utf-8 -*-
"""
SOC 2 Type II Preparation API - SEC-009 Phase 10

FastAPI router aggregation for the SOC 2 Type II audit preparation platform.
Provides RESTful endpoints for all SOC 2 preparation functionality including:

- Self-assessment execution and readiness scoring
- Evidence collection, packaging, and validation
- Control testing execution and reporting
- Auditor portal with secure access management
- Findings tracking and remediation workflows
- Management attestation and digital signatures
- Audit project management and milestones
- Dashboard metrics and analytics

All endpoints are protected by the GreenLang RBAC system (SEC-002) with
fine-grained permission mappings defined in route_protector.py.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.soc2_preparation.api import soc2_router
    >>> app = FastAPI()
    >>> app.include_router(soc2_router)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import Sub-Routers
# ---------------------------------------------------------------------------

from greenlang.infrastructure.soc2_preparation.api.assessment_routes import (
    router as assessment_router,
)
from greenlang.infrastructure.soc2_preparation.api.evidence_routes import (
    router as evidence_router,
)
from greenlang.infrastructure.soc2_preparation.api.testing_routes import (
    router as testing_router,
)
from greenlang.infrastructure.soc2_preparation.api.portal_routes import (
    router as portal_router,
)
from greenlang.infrastructure.soc2_preparation.api.findings_routes import (
    router as findings_router,
)
from greenlang.infrastructure.soc2_preparation.api.attestation_routes import (
    router as attestation_router,
)
from greenlang.infrastructure.soc2_preparation.api.project_routes import (
    router as project_router,
)
from greenlang.infrastructure.soc2_preparation.api.dashboard_routes import (
    router as dashboard_router,
)

# ---------------------------------------------------------------------------
# Main SOC 2 Router
# ---------------------------------------------------------------------------

soc2_router = APIRouter(
    prefix="/api/v1/soc2",
    tags=["soc2"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Permission denied"},
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
soc2_router.include_router(assessment_router)
soc2_router.include_router(evidence_router)
soc2_router.include_router(testing_router)
soc2_router.include_router(portal_router)
soc2_router.include_router(findings_router)
soc2_router.include_router(attestation_router)
soc2_router.include_router(project_router)
soc2_router.include_router(dashboard_router)

logger.info("SOC 2 API router initialized with 8 sub-routers")

__all__ = [
    "soc2_router",
    # Individual routers for direct use
    "assessment_router",
    "evidence_router",
    "testing_router",
    "portal_router",
    "findings_router",
    "attestation_router",
    "project_router",
    "dashboard_router",
]
