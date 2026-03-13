# -*- coding: utf-8 -*-
"""
AGENT-EUDR-025: Risk Mitigation Advisor - API Package

FastAPI router package providing 46 REST endpoints for EUDR risk mitigation
advisory operations including ML-powered strategy recommendation, remediation
plan management, supplier capacity building, mitigation measure library,
effectiveness tracking, adaptive monitoring, cost-benefit optimization,
stakeholder collaboration, and compliance reporting.

Route Modules:
    - strategy_routes: ML-powered strategy selection (recommend, list, detail, select, explain)
    - plan_routes: Remediation plan lifecycle (create, list, detail, update, status, clone,
                   gantt, milestones, evidence)
    - capacity_routes: Supplier capacity building (enroll, enrollments, progress, scorecard)
    - library_routes: Mitigation measure library (search, compare, packages, detail)
    - effectiveness_routes: Effectiveness tracking (plan, supplier, portfolio, roi)
    - monitoring_routes: Adaptive management (triggers, acknowledge, dashboard, drift)
    - optimization_routes: Cost-benefit optimization (run, result, pareto, sensitivity)
    - collaboration_routes: Stakeholder collaboration (messages, tasks, supplier-portal)
    - reporting_routes: Compliance reporting (generate, list, download, dds-section)
    - router: Main router registration with /v1/eudr-rma prefix + health + stats

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with 21 eudr-rma:* permissions
    - Rate limiting via per-endpoint RateLimiter instances
    - API key fallback via X-API-Key header

Engines Mapped:
    1. StrategySelectionEngine     -> strategy_routes
    2. RemediationPlanDesignEngine -> plan_routes
    3. CapacityBuildingManager     -> capacity_routes
    4. MeasureLibraryEngine        -> library_routes
    5. EffectivenessTrackingEngine -> effectiveness_routes
    6. ContinuousMonitoringEngine  -> monitoring_routes
    7. CostBenefitOptimizer        -> optimization_routes
    8. StakeholderCollaboration    -> collaboration_routes
    9. ReportGenerator (service)   -> reporting_routes

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
Status: Production Ready
"""

from greenlang.agents.eudr.risk_mitigation_advisor.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
