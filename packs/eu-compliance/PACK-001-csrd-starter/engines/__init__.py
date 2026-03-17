# -*- coding: utf-8 -*-
"""
PACK-001 Calculation Engines
=============================

Provides the CSRD Starter Pack calculation engine layer, wrapping
GreenLang MRV agents via the MRVBridge for ESRS E1 climate metrics.

Modules:
    csrd_calculation_engine: Primary calculation engine bridging MRV agents
"""

try:
    from .csrd_calculation_engine import CSRDCalculationEngine
except ImportError:
    CSRDCalculationEngine = None  # type: ignore[assignment, misc]

__all__ = ["CSRDCalculationEngine"]
