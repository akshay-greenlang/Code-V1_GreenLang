# -*- coding: utf-8 -*-
"""
PII Service API - SEC-011

FastAPI router exposing PII detection, redaction, tokenization, and
management endpoints for the unified PII service.

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

from greenlang.infrastructure.pii_service.api.pii_routes import pii_router

__all__ = [
    "pii_router",
]
