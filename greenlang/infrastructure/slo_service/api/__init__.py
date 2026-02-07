# -*- coding: utf-8 -*-
"""
SLO Service REST API - OBS-005

Re-exports the SLO API router for inclusion in the main application.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from greenlang.infrastructure.slo_service.api.router import slo_router
except ImportError:
    slo_router = None  # type: ignore[assignment]
    logger.warning("slo_router not available")

__all__ = ["slo_router"]
