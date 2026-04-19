# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER API Module

FastAPI REST endpoints for steam trap monitoring.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from .routes import router, create_app

__all__ = ["router", "create_app"]
