# -*- coding: utf-8 -*-
"""
Chain of Custody REST API - AGENT-EUDR-009

FastAPI router package providing 37 REST endpoints for EUDR chain of
custody operations including custody events, batch lifecycle, CoC model
management, mass balance ledger, transformations, documents, verification,
reports, batch jobs, and health.

Route Modules:
    - event_routes: Custody event recording, chain queries, amendments
    - batch_routes: Batch CRUD, split, merge, blend, genealogy, search
    - model_routes: CoC model assignment, validation, compliance scoring
    - balance_routes: Mass balance input/output, reconciliation, history;
                      transformations; document linking and validation
    - verification_routes: Chain verification, batch verification
    - report_routes: Article 9 traceability and mass balance reports
    - router: Main router with batch jobs and health endpoints

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-coc:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Chain of Custody)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from greenlang.agents.eudr.chain_of_custody.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
