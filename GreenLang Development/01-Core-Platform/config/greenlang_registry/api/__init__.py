"""
FastAPI application package for GreenLang Agent Registry.

Provides RESTful API endpoints for agent lifecycle management.
"""

from greenlang_registry.api.app import create_app, app

__all__ = ["create_app", "app"]
