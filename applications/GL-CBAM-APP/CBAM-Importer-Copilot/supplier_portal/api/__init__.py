# -*- coding: utf-8 -*-
"""
CBAM Supplier Portal API - Router Package

Provides FastAPI routers for the CBAM Supplier Portal, enabling
third-country suppliers to register, manage installations, and
submit verified emissions data to EU importers.

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from supplier_portal.api.supplier_routes import router as supplier_router

__all__ = ["supplier_router"]
