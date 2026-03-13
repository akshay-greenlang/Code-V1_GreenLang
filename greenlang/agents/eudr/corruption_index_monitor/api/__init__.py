# -*- coding: utf-8 -*-
"""
AGENT-EUDR-019: Corruption Index Monitor - API Package

FastAPI router package providing 40+ REST endpoints for EUDR corruption index
monitoring operations including CPI score monitoring, WGI analysis, bribery risk
assessment, institutional quality scoring, trend analysis, deforestation-corruption
correlation, alert management, and compliance impact assessment.

Route Modules:
    - cpi_routes: CPI monitoring (score, history, rankings, regional, batch, summary)
    - wgi_routes: WGI analysis (indicators, history, dimension, compare, rankings)
    - bribery_routes: Bribery risk (assess, profile, sectors, high-risk, sector-analysis)
    - institutional_routes: Institutional quality (quality, governance, assess, forest, compare)
    - trend_routes: Trend analysis (analyze, trajectory, prediction, improving, deteriorating)
    - correlation_routes: Correlation analysis (analyze, deforestation, regression, heatmap, causal)
    - alert_routes: Alert management (list, detail, configure, acknowledge, summary)
    - compliance_routes: Compliance impact (assess-impact, country, dd-recommendations, classifications)
    - router: Main router registration with /v1/eudr-cim prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-corruption-index:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from greenlang.agents.eudr.corruption_index_monitor.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
