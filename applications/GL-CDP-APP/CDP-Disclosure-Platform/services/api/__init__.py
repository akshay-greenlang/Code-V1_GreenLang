"""
GL-CDP-APP REST API Package

CDP Climate Change Disclosure Platform for organizational CDP questionnaire
management, scoring simulation, gap analysis, benchmarking, supply chain
engagement, transition plan building, and submission reporting.

Provides 80+ endpoints across 10 routers covering the full CDP disclosure
lifecycle: questionnaires, responses, scoring, gap analysis, benchmarking,
supply chain, transition plans, reporting, dashboard, and settings.

Routes:
    - questionnaire_routes: Questionnaire structure, modules, questions
    - response_routes: Response CRUD, workflow, versioning, evidence
    - scoring_routes: CDP scoring simulation, what-if, A-level checks
    - gap_analysis_routes: Gap identification, recommendations, uplift
    - benchmarking_routes: Sector/regional benchmarking, peer comparison
    - supply_chain_routes: Supplier engagement, emissions aggregation
    - transition_plan_routes: 1.5C transition plan, milestones, SBTi
    - reporting_routes: Report generation (PDF/Excel/XML), submission
    - dashboard_routes: Dashboard metrics, readiness, timeline, activity
    - settings_routes: Organization settings, team, MRV integrations
"""

from api.questionnaire_routes import router as questionnaire_router
from api.response_routes import router as response_router
from api.scoring_routes import router as scoring_router
from api.gap_analysis_routes import router as gap_analysis_router
from api.benchmarking_routes import router as benchmarking_router
from api.supply_chain_routes import router as supply_chain_router
from api.transition_plan_routes import router as transition_plan_router
from api.reporting_routes import router as reporting_router
from api.dashboard_routes import router as dashboard_router
from api.settings_routes import router as settings_router

__all__ = [
    "questionnaire_router",
    "response_router",
    "scoring_router",
    "gap_analysis_router",
    "benchmarking_router",
    "supply_chain_router",
    "transition_plan_router",
    "reporting_router",
    "dashboard_router",
    "settings_router",
]
