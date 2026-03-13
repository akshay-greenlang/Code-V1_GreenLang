# -*- coding: utf-8 -*-
"""
Capacity Building Training Modules - AGENT-EUDR-025

7 commodities x 22 modules = 154 total training modules organized
across 4 progressive tiers for supplier capacity building.

Tiers:
    Tier 1 (Awareness): 4 modules per commodity - Basic EUDR requirements
    Tier 2 (Basic Compliance): 8 modules per commodity - Data collection skills
    Tier 3 (Advanced Practices): 6 modules per commodity - Sustainable production
    Tier 4 (Leadership): 4 modules per commodity - Certification readiness

Commodities:
    cattle, cocoa, coffee, palm_oil, rubber, soya, wood

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

from typing import Dict, List

COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil",
    "rubber", "soya", "wood",
]

TIERS: Dict[int, Dict[str, Any]] = {
    1: {"name": "Awareness", "modules_count": 4},
    2: {"name": "Basic Compliance", "modules_count": 8},
    3: {"name": "Advanced Practices", "modules_count": 6},
    4: {"name": "Leadership", "modules_count": 4},
}

MODULES_PER_COMMODITY: int = 22
TOTAL_MODULES: int = len(COMMODITIES) * MODULES_PER_COMMODITY  # 154


def get_total_modules() -> int:
    """Return total number of training modules."""
    return TOTAL_MODULES


def get_modules_per_commodity() -> int:
    """Return modules per commodity."""
    return MODULES_PER_COMMODITY


def get_commodities() -> List[str]:
    """Return list of supported commodities."""
    return list(COMMODITIES)
