# -*- coding: utf-8 -*-
"""
AGENT-EUDR-020: Deforestation Alert System - API Package

FastAPI router package providing 30+ REST endpoints for EUDR deforestation
alert system operations including satellite change detection, alert management,
severity classification, spatial buffer monitoring, EUDR cutoff date
verification, historical baseline comparison, alert workflow management,
and compliance impact assessment.

Route Modules:
    - satellite_routes: Satellite change detection (detect, scan, sources, imagery)
    - alert_routes: Alert management (list, detail, create, batch, summary, statistics)
    - severity_routes: Severity classification (classify, reclassify, thresholds, distribution)
    - buffer_routes: Spatial buffer monitoring (create, update, check, violations, zones)
    - cutoff_routes: Cutoff date verification (verify, batch-verify, evidence, timeline)
    - baseline_routes: Historical baselines (establish, compare, update, coverage)
    - workflow_routes: Alert workflow (triage, assign, investigate, resolve, escalate, sla)
    - compliance_routes: Compliance impact (assess, affected-products, recommendations, remediation)
    - router: Main router registration with /v1/eudr-das prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-deforestation-alert:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from greenlang.agents.eudr.deforestation_alert_system.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
