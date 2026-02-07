# -*- coding: utf-8 -*-
"""
WAF Management API - SEC-010

FastAPI router for WAF rule management, attack detection,
and DDoS protection endpoints.

This module provides REST endpoints for:
    - WAF rule CRUD operations
    - Rule testing and deployment
    - Attack detection and mitigation
    - Traffic metrics and monitoring
    - Shield Advanced status

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.waf_management.api import waf_router
    >>> app = FastAPI()
    >>> app.include_router(waf_router, prefix="/api/v1/secops/waf")
"""

from greenlang.infrastructure.waf_management.api.waf_routes import waf_router

__all__ = ["waf_router"]
