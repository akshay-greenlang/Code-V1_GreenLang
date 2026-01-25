"""
GreenLang Agent Factory - FastAPI Application

This package contains the FastAPI application with:
- API routers for agents, executions, search
- Middleware for auth, rate limiting, validation
- Application lifecycle management
"""

from app.main import create_app

__all__ = ["create_app"]
