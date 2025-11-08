"""
API routes for GreenLang.

This module provides REST API endpoints for various features including
dashboards, workflows, agents, and analytics.
"""

from greenlang.api.routes.dashboards import router as dashboards_router

__all__ = ["dashboards_router"]
