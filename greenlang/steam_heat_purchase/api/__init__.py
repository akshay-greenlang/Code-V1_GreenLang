# -*- coding: utf-8 -*-
"""API package for AGENT-MRV-011 Steam/Heat Purchase Agent.

Provides FastAPI router with 20 REST endpoints at /api/v1/steam-heat-purchase.
"""

try:
    from greenlang.steam_heat_purchase.api.router import router
except ImportError:
    router = None

__all__ = ["router"]
