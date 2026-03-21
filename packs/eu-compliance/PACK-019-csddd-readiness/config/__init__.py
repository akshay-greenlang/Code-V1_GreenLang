# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Configuration Module
=========================================================

Provides pack-level configuration for the CSDDD Readiness Pack,
including company scope settings, sector types, impact assessment
parameters, and engine-specific configuration models.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .pack_config import PackConfig
except ImportError as e:
    PackConfig = None  # type: ignore[assignment,misc]
    logger.debug("PackConfig not available: %s", e)

__all__ = [
    "PackConfig",
]
