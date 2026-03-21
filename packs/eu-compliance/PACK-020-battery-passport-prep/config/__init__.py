# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Configuration Module
===============================================================

Provides pack-level configuration for the Battery Passport Prep Pack,
including battery category settings, chemistry types, lifecycle stages,
and engine-specific configuration models.
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
