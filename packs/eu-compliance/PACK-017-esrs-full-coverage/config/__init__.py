# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Configuration Module
=========================================================

Provides the configuration system for the ESRS Full Coverage Pack
including enums, Pydantic v2 models, preset loading, and pack config.

Author: GreenLang Team
Version: 17.0.0
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .pack_config import (
        PackConfig,
        ESRSFullCoverageConfig,
        ESRSStandard,
        MaterialityStatus,
    )
except ImportError as e:
    logger.debug("pack_config not available: %s", e)

__all__ = [
    "PackConfig",
    "ESRSFullCoverageConfig",
    "ESRSStandard",
    "MaterialityStatus",
]
