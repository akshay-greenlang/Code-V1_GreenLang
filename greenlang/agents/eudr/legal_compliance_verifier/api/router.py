# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-023 Legal Compliance Verifier API

Aggregates all route modules under the ``/v1/eudr-lcv`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (37 endpoints + health = 38):
    framework_routes:     5 endpoints (POST register, GET list, GET detail, PUT update, POST search)
    document_routes:      5 endpoints (POST verify, GET list, GET detail, POST validity-check, GET expiring)
    certification_routes: 4 endpoints (POST validate, GET list, GET detail, POST eudr-equivalence)
    red_flag_routes:      4 endpoints (POST detect, GET list, GET detail, PUT suppress)
    compliance_routes:    5 endpoints (POST assess, POST check-category, GET list, GET detail, GET history)
    audit_routes:         4 endpoints (POST ingest, GET list, GET findings, PUT corrective-actions)
    report_routes:        4 endpoints (POST generate, GET list, GET download, POST schedule)
    batch_routes:         3 endpoints (POST assess, POST verify, GET status)
    + health check:       1 endpoint (GET health)
    -------
    Total: 35 unique endpoints + 2 batch POST + 1 health = 38

Auth & RBAC (20 permissions):
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-lcv:* permissions via SEC-002.

    Permissions:
        eudr-lcv:framework:create    eudr-lcv:framework:read     eudr-lcv:framework:update
        eudr-lcv:document:create     eudr-lcv:document:read
        eudr-lcv:certification:create eudr-lcv:certification:read
        eudr-lcv:red-flag:create     eudr-lcv:red-flag:read      eudr-lcv:red-flag:update
        eudr-lcv:compliance:create   eudr-lcv:compliance:read
        eudr-lcv:audit:create        eudr-lcv:audit:read         eudr-lcv:audit:update
        eudr-lcv:report:create       eudr-lcv:report:read
        eudr-lcv:batch:create        eudr-lcv:batch:read
        eudr-lcv:*                   (wildcard, grants all above)

Rate Limiting:
    Standard (100/min): GET list, GET detail, POST search, POST validity-check,
                        POST check-category, POST eudr-equivalence
    Write (30/min):     POST register, PUT update, POST verify, PUT suppress,
                        POST ingest, PUT corrective-actions, POST schedule
    Heavy (10/min):     POST detect, POST assess, POST batch/assess, POST batch/verify
    Export (5/min):     POST reports/generate

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.legal_compliance_verifier.api.framework_routes import (
    router as framework_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.document_routes import (
    router as document_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.certification_routes import (
    router as certification_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.red_flag_routes import (
    router as red_flag_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.compliance_routes import (
    router as compliance_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.audit_routes import (
    router as audit_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.batch_routes import (
    router as batch_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-lcv prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-lcv",
    tags=["eudr-legal-compliance-verifier"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(framework_router)
router.include_router(document_router)
router.include_router(certification_router)
router.include_router(red_flag_router)
router.include_router(compliance_router)
router.include_router(audit_router)
router.include_router(report_router)
router.include_router(batch_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    tags=["health"],
    summary="Legal Compliance Verifier health check",
    description="Returns service health status for AGENT-EUDR-023.",
)
async def health_check() -> dict:
    """Return health status for the Legal Compliance Verifier agent.

    Returns:
        Dictionary with status, agent ID, and component name.
    """
    return {
        "status": "healthy",
        "agent_id": "GL-EUDR-LCV-023",
        "agent": "EUDR-023",
        "component": "legal-compliance-verifier",
        "version": "1.0.0",
    }


def get_router() -> APIRouter:
    """Return the EUDR Legal Compliance Verifier API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.legal_compliance_verifier.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all legal compliance verifier endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
