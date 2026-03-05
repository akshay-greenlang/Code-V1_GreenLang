"""
GL-TCFD-APP REST API Package

TCFD (Task Force on Climate-related Financial Disclosures) Platform for
climate-related financial disclosure management across the four TCFD pillars:
Governance, Strategy, Risk Management, and Metrics & Targets.  Includes
scenario analysis (IEA / NGFS), physical and transition risk assessment,
financial impact modeling, ISSB/IFRS S2 cross-walk, and gap analysis.

Provides 65+ endpoints across 14 routers covering the full TCFD disclosure
lifecycle: governance assessments, strategy mapping, scenario modeling,
physical risk, transition risk, opportunities, financial impact, risk
management, metrics & targets, disclosure generation, dashboard, gap
analysis, ISSB cross-walk, and platform settings.

Routes:
    - governance_routes: Governance assessments, roles, maturity, competency
    - strategy_routes: Climate risks, opportunities, business & value chain impact
    - scenario_routes: Pre-built & custom scenarios, analysis, sensitivity, Monte Carlo
    - physical_risk_routes: Asset registration, hazard assessment, risk mapping
    - transition_risk_routes: Policy/technology/market/reputation risk assessment
    - opportunity_routes: Climate opportunity pipeline, sizing, ROI, financing
    - financial_routes: Income/balance sheet/cash flow impacts, NPV, IRR, MACC
    - risk_management_routes: Risk records, responses, heat map, ERM integration
    - metrics_routes: Cross-industry & industry metrics, targets, SBTi, benchmarking
    - disclosure_routes: Disclosure CRUD, section generation, compliance, export
    - dashboard_routes: Executive KPI summary, scenario comparison, progress
    - gap_routes: Gap assessment, pillar scores, action plan, recommendations
    - issb_routes: TCFD-to-IFRS S2 mapping, dual compliance, migration pathway
    - settings_routes: Organization settings, currencies, sectors, hazard types
"""

from .governance_routes import router as governance_router
from .strategy_routes import router as strategy_router
from .scenario_routes import router as scenario_router
from .physical_risk_routes import router as physical_risk_router
from .transition_risk_routes import router as transition_risk_router
from .opportunity_routes import router as opportunity_router
from .financial_routes import router as financial_router
from .risk_management_routes import router as risk_management_router
from .metrics_routes import router as metrics_router
from .disclosure_routes import router as disclosure_router
from .dashboard_routes import router as dashboard_router
from .gap_routes import router as gap_router
from .issb_routes import router as issb_router
from .settings_routes import router as settings_router

__all__ = [
    "governance_router",
    "strategy_router",
    "scenario_router",
    "physical_risk_router",
    "transition_risk_router",
    "opportunity_router",
    "financial_router",
    "risk_management_router",
    "metrics_router",
    "disclosure_router",
    "dashboard_router",
    "gap_router",
    "issb_router",
    "settings_router",
]
