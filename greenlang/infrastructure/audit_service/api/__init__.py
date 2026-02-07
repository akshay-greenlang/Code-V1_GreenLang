# -*- coding: utf-8 -*-
"""
Audit Service REST API - SEC-005

FastAPI routers for audit event querying, searching, statistics,
real-time streaming, compliance reporting, and data export.

Routers:
    events_router  - /api/v1/audit/events (list, get)
    search_router  - /api/v1/audit/search (advanced search)
    stats_router   - /api/v1/audit/stats, /timeline, /hotspots
    stream_router  - /api/v1/audit/events/stream (WebSocket)
    report_router  - /api/v1/audit/reports (SOC2, ISO27001, GDPR)
    export_router  - /api/v1/audit/export (CSV, JSON, Parquet)

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[misc, assignment]

# Import sub-routers
try:
    from greenlang.infrastructure.audit_service.api.events_routes import events_router
except ImportError:
    events_router = None

try:
    from greenlang.infrastructure.audit_service.api.search_routes import search_router
except ImportError:
    search_router = None

try:
    from greenlang.infrastructure.audit_service.api.stats_routes import stats_router
except ImportError:
    stats_router = None

try:
    from greenlang.infrastructure.audit_service.api.stream_routes import stream_router
except ImportError:
    stream_router = None

try:
    from greenlang.infrastructure.audit_service.api.report_routes import report_router
except ImportError:
    report_router = None

try:
    from greenlang.infrastructure.audit_service.api.export_routes import export_router
except ImportError:
    export_router = None

# Combined router
if FASTAPI_AVAILABLE:
    audit_router = APIRouter()

    if events_router:
        audit_router.include_router(events_router)
    if search_router:
        audit_router.include_router(search_router)
    if stats_router:
        audit_router.include_router(stats_router)
    if stream_router:
        audit_router.include_router(stream_router)
    if report_router:
        audit_router.include_router(report_router)
    if export_router:
        audit_router.include_router(export_router)
else:
    audit_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - audit_router is None")

__all__ = [
    "audit_router",
    "events_router",
    "search_router",
    "stats_router",
    "stream_router",
    "report_router",
    "export_router",
]
