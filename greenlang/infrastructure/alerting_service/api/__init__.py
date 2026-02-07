# -*- coding: utf-8 -*-
"""
Alerting Service REST API - OBS-004

Re-exports the alerts API router for inclusion in the main application.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from greenlang.infrastructure.alerting_service.api.router import alerts_router
except ImportError:
    alerts_router = None  # type: ignore[assignment]
    logger.warning("alerts_router not available")

__all__ = ["alerts_router"]
