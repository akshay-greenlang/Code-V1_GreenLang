# -*- coding: utf-8 -*-
"""
Satellite Monitoring REST API - AGENT-EUDR-003

FastAPI router package providing 28 REST endpoints for EUDR satellite
monitoring operations including imagery search and download, spectral
index calculation, baseline establishment, change detection, multi-source
fusion, continuous monitoring schedules, alert management, evidence
package generation, and batch satellite analysis.

Route Modules:
    - imagery_routes: Satellite scene search, download, and availability
    - analysis_routes: Spectral index, baseline, change detection, fusion
    - monitoring_routes: Schedule CRUD and manual execution
    - alert_routes: Alert listing, detail, acknowledgement, and summary
    - evidence_routes: Evidence package generation and retrieval
    - batch_routes: Batch analysis submission, progress, and cancellation
    - router: Main router registration with /v1/eudr-sat prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-satellite:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
Status: Production Ready
"""

from greenlang.agents.eudr.satellite_monitoring.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
