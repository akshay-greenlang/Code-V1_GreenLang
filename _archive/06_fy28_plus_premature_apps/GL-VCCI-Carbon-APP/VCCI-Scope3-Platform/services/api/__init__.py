# -*- coding: utf-8 -*-
"""
API Routes Package

FastAPI route modules for the VCCI Scope 3 Platform.
Includes v1.1 enhancements: uncertainty, CDP, compliance, settings.
"""

from .uncertainty_routes import router as uncertainty_router
from .cdp_routes import router as cdp_router
from .compliance_routes import router as compliance_router
from .settings_routes import router as settings_router

__all__ = [
    "uncertainty_router",
    "cdp_router",
    "compliance_router",
    "settings_router",
]
