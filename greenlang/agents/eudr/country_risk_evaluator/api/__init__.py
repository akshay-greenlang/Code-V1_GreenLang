# -*- coding: utf-8 -*-
"""
Country Risk Evaluator API Package - AGENT-EUDR-016

FastAPI REST API for the EUDR Country Risk Evaluator Agent providing
programmatic access to country risk scoring, commodity risk analysis,
deforestation hotspot detection, governance evaluation, due diligence
classification, trade flow analysis, report generation, and regulatory
update tracking.

Prefix: /api/v1/eudr-cre
Permission prefix: eudr-cre:*

Sub-routers (8):
    - country_routes: Country risk assessment (5 endpoints)
    - commodity_routes: Commodity risk analysis (4 endpoints)
    - hotspot_routes: Deforestation hotspot detection (5 endpoints)
    - governance_routes: Governance evaluation (4 endpoints)
    - due_diligence_routes: Due diligence classification (5 endpoints)
    - trade_flow_routes: Trade flow analysis (5 endpoints)
    - report_routes: Report generation (5 endpoints)
    - regulatory_routes: Regulatory update tracking (4 endpoints)
    + health (1 endpoint) = 38 total

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.agents.eudr.country_risk_evaluator.api.router import (
    get_router,
    router,
)

__all__ = [
    "get_router",
    "router",
]
