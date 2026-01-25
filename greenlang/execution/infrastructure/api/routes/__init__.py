"""
GreenLang REST API Routes

This module provides domain-specific REST API routes for the GreenLang platform.
All routes follow OpenAPI 3.0 specifications with comprehensive error handling,
authentication, and rate limiting.

Routes:
    - emissions_routes: Emission calculation endpoints
    - agents_routes: Agent management endpoints
    - jobs_routes: Job queue and status endpoints
    - compliance_routes: Compliance report endpoints
"""

from greenlang.infrastructure.api.routes.emissions_routes import emissions_router
from greenlang.infrastructure.api.routes.agents_routes import agents_router
from greenlang.infrastructure.api.routes.jobs_routes import jobs_router
from greenlang.infrastructure.api.routes.compliance_routes import compliance_router

__all__ = [
    "emissions_router",
    "agents_router",
    "jobs_router",
    "compliance_router",
]
