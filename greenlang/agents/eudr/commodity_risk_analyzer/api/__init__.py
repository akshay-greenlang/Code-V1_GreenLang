# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer REST API - AGENT-EUDR-018

FastAPI router package providing 40+ REST endpoints for EUDR commodity risk
analysis operations including commodity profiling, derived product traceability,
price volatility monitoring, production forecasting, substitution risk detection,
regulatory compliance checking, due diligence workflow management, and portfolio
risk aggregation.

Route Modules:
    - commodity_routes: Commodity profiling (profile, batch, risk, history, compare, summary)
    - derived_product_routes: Derived product analysis (analyze, chain, risk, mapping, trace)
    - price_routes: Price and market data (current, history, volatility, disruptions, forecast)
    - production_routes: Production forecasting (forecast, yield, climate-impact, seasonal, summary)
    - substitution_routes: Substitution risk (detect, history, alerts, verify, patterns)
    - regulatory_routes: Regulatory compliance (requirements, check, penalty, updates, docs)
    - due_diligence_routes: DD workflows (initiate, status, evidence, pending, complete)
    - portfolio_routes: Portfolio aggregation (analyze, concentration, diversification, summary)
    - router: Main router registration with /v1/eudr-cra prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-commodity-risk:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from greenlang.agents.eudr.commodity_risk_analyzer.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
