# -*- coding: utf-8 -*-
"""
API package for AGENT-MRV-009 Scope 2 Location-Based Emissions Agent.

Provides FastAPI router with 20 REST endpoints at /api/v1/scope2-location.
"""

try:
    from greenlang.agents.mrv.scope2_location.api.router import router
except ImportError:
    router = None

__all__ = ["router"]
