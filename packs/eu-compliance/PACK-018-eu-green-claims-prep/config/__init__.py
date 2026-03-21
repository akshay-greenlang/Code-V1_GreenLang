# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Configuration Module
=============================================================

Provides pack-level configuration for the EU Green Claims Prep Pack,
including claim scope settings, communication channel configuration,
and engine-specific parameters.
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
