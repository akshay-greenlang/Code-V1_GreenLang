# -*- coding: utf-8 -*-
"""API package for AGENT-MRV-010 Scope 2 Market-Based Emissions Agent.

Provides FastAPI router with 20 REST endpoints at /api/v1/scope2-market.
"""

try:
    from greenlang.scope2_market.api.router import router
except ImportError:
    router = None

__all__ = ["router"]
