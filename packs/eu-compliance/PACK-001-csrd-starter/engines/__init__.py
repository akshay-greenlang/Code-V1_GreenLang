# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Engines
======================================

Provides the CSRD Starter Pack calculation engine layer, wrapping
GreenLang MRV agents via the MRVBridge for ESRS E1 climate metrics.

Engines:
    1. CSRDCalculationEngine  - Primary calculation engine bridging MRV agents

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-001 CSRD Starter Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-001"
__pack_name__: str = "CSRD Starter Pack"
__engines_count__: int = 1

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: CSRD Calculation Engine
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "CSRDCalculationEngine",
]

try:
    from .csrd_calculation_engine import (  # noqa: F401
        CSRDCalculationEngine,
    )
    _loaded_engines.append("CSRDCalculationEngine")
except ImportError as e:
    logger.debug("Engine 1 (CSRDCalculationEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Module exports
# ===================================================================
_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_ENGINE_1_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-001 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
