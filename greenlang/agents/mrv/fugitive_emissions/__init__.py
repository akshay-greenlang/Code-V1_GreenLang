# -*- coding: utf-8 -*-
"""
GreenLang Fugitive Emissions Agent - AGENT-MRV-005 (GL-MRV-SCOPE1-005)

Scope 1 fugitive emission estimation covering equipment leaks (LDAR),
coal mine methane, wastewater treatment, pneumatic devices, tank losses,
and direct measurement integration.

Engines:
    1. FugitiveSourceDatabaseEngine  - Source types, emission factors, gas compositions
    2. EmissionCalculatorEngine      - 5 calculation methods (avg EF, screening, correlation,
                                       engineering estimates, direct measurement)
    3. LeakDetectionEngine           - LDAR survey scheduling, leak tracking, repair management

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "FugitiveSourceDatabaseEngine",
    "EmissionCalculatorEngine",
    "LeakDetectionEngine",
]

VERSION: str = "1.0.0"
