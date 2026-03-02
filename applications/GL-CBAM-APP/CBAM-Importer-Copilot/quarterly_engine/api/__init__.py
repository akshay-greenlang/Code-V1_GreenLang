# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Quarterly Engine API v1.1

FastAPI router for CBAM quarterly report management, deadline tracking,
amendment handling, and notification configuration.

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

from .quarterly_routes import router as quarterly_router

__version__ = "1.1.0"
__all__ = ["quarterly_router"]
