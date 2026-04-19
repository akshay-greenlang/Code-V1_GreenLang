# -*- coding: utf-8 -*-
"""
API routes for GreenLang.

This module provides REST API endpoints for various features including
emission factors, editions, calculations, dashboards, workflows, agents,
and analytics.

Organization:
- factors: Emission factor query endpoints
- editions: Factor catalog edition management
- calculations: Emission calculation endpoints
- health: Health check and monitoring endpoints
- dashboards: Dashboard and analytics endpoints
- marketplace: Marketplace endpoints
"""

from greenlang.integration.api.routes.factors import router as factors_router
from greenlang.integration.api.routes.editions import router as editions_router
from greenlang.integration.api.routes.calculations import router as calculations_router
from greenlang.integration.api.routes.health import router as health_router
from greenlang.integration.api.routes.dashboards import router as dashboards_router
from greenlang.integration.api.routes.marketplace import router as marketplace_router

__all__ = [
    "factors_router",
    "editions_router",
    "calculations_router",
    "health_router",
    "dashboards_router",
    "marketplace_router",
]
