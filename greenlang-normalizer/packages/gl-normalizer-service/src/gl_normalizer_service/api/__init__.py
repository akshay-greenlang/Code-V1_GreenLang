"""
API module for GL Normalizer Service.

This module contains the FastAPI route handlers, Pydantic models for
request/response validation, and dependency injection utilities.

Submodules:
    routes: API endpoint handlers
    models: Pydantic request/response models
    deps: Dependency injection utilities
"""

from gl_normalizer_service.api.routes import router

__all__ = ["router"]
