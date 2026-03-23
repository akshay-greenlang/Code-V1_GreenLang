# -*- coding: utf-8 -*-
"""
API package for AGENT-MRV-013 Dual Reporting Reconciliation Agent.

Provides FastAPI router with 16 REST endpoints at /api/v1/dual-reporting.
"""

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.api.router import router
except ImportError:
    router = None

__all__ = ["router"]
