# -*- coding: utf-8 -*-
"""
Mass Balance Calculator REST API - AGENT-EUDR-011

FastAPI router package providing 34 REST endpoints for EUDR mass balance
calculator operations including double-entry ledger management, credit
period lifecycle, conversion factor validation, overdraft detection,
loss/waste tracking, reconciliation with anomaly detection, and
multi-facility consolidation reporting.

Route Modules:
    - ledger_routes: Ledger CRUD, entries, bulk import, balance, history, search
    - period_routes: Credit period create, detail, extend, rollover, active listing
    - factor_routes: Conversion factor validation, reference lookup, custom, history
    - overdraft_routes: Overdraft check, alerts, forecast, exemption, history
    - loss_routes: Loss recording, listing, validation, trend analysis
    - reconciliation_routes: Reconciliation run, result, sign-off, history
    - consolidation_routes: Report generation, groups, dashboard, download
    - router: Main router with batch job and health endpoints

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-mbc:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011 (Mass Balance Calculator)
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from greenlang.agents.eudr.mass_balance_calculator.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
