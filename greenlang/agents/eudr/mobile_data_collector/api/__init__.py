# -*- coding: utf-8 -*-
"""
Mobile Data Collector API Package - AGENT-EUDR-015

FastAPI REST API layer for the EUDR Mobile Data Collector Agent
providing endpoints for offline form management, GPS capture, photo
evidence, sync management, form templates, digital signatures, data
packages, and device fleet management.

Prefix: /api/v1/eudr-mdc
Tags: eudr-mobile-data-collector
Endpoints: 70+ across 8 sub-routers + health

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-mdc:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

logger = logging.getLogger(__name__)


def get_router() -> APIRouter:
    """Return the EUDR Mobile Data Collector API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.mobile_data_collector.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all mobile data collector endpoints.
    """
    from greenlang.agents.eudr.mobile_data_collector.api.router import (
        router,
    )

    return router


__all__ = [
    "get_router",
]
