# -*- coding: utf-8 -*-
"""
CBAM Certificate Engine API - Router Package

Provides FastAPI routers for CBAM certificate obligation calculation,
EU ETS price management, free allocation benchmarks, and carbon price
deduction tracking.

Per EU CBAM Regulation 2023/956 Articles 21-31.

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from .certificate_routes import router as certificate_router

__all__ = ["certificate_router"]
