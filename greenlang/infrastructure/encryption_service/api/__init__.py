# -*- coding: utf-8 -*-
"""
Encryption REST API - SEC-003: Encryption at Rest

FastAPI router providing REST endpoints for the encryption service.
Supports encrypt/decrypt operations, key management, and audit log access.

Provides:
    - Data encryption/decryption endpoints
    - Key rotation and management
    - Audit log retrieval
    - Service health status

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.encryption_service.api import encryption_router
    >>> app = FastAPI()
    >>> app.include_router(encryption_router)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from greenlang.infrastructure.encryption_service.api.encryption_routes import (
        encryption_router,
    )

    __all__ = ["encryption_router"]

except ImportError as e:
    logger.warning(
        "Failed to import encryption_router: %s - router will be None",
        e,
    )
    encryption_router = None  # type: ignore[assignment]
    __all__ = []
