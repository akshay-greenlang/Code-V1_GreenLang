# -*- coding: utf-8 -*-
"""
IUCN Category Reference Data - AGENT-EUDR-022

Authoritative reference data for IUCN protected area management categories
including risk scores, commodity production allowance, and default buffer
zone radii per PRD Section 4.3.

IUCN Categories:
    Ia  - Strict Nature Reserve:       100 (CRITICAL) - Absolutely prohibited
    Ib  - Wilderness Area:             100 (CRITICAL) - Absolutely prohibited
    II  - National Park:                95 (CRITICAL) - Prohibited
    III - Natural Monument:             85 (HIGH) - Prohibited
    IV  - Habitat/Species Mgmt Area:    80 (HIGH) - Generally prohibited
    V   - Protected Landscape:          50 (MEDIUM) - Conditionally allowed
    VI  - Sustainable Use:              25 (LOW) - Allowed under plan
    NR  - Not Reported:                 75 (HIGH default) - Uncertain

Example:
    >>> from greenlang.agents.eudr.protected_area_validator.reference_data.iucn_categories import (
    ...     get_iucn_risk_score,
    ... )
    >>> score = get_iucn_risk_score("Ia")
    >>> assert score == 100

Author: GreenLang Platform Team
Date: March 2026
"""

from decimal import Decimal
from typing import Any, Dict

# ---------------------------------------------------------------------------
# IUCN category base risk scores (PRD Section 4.3)
# ---------------------------------------------------------------------------

IUCN_CATEGORY_SCORES: Dict[str, int] = {
    "Ia": 100,
    "Ib": 100,
    "II": 95,
    "III": 85,
    "IV": 80,
    "V": 50,
    "VI": 25,
    "NR": 75,
}

# ---------------------------------------------------------------------------
# IUCN category full definitions
# ---------------------------------------------------------------------------

IUCN_CATEGORY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Ia": {
        "name": "Strict Nature Reserve",
        "description": (
            "Strictly protected for biodiversity and possibly geological/"
            "geomorphological features, where human visitation, use and "
            "impacts are strictly controlled and limited."
        ),
        "commodity_production_allowed": False,
        "commodity_production_note": "Absolutely prohibited",
        "risk_score": 100,
        "risk_level": "critical",
        "default_buffer_km": Decimal("10"),
        "examples": [
            "Galápagos Marine Reserve (Ecuador)",
            "Białowieża National Park strict zone (Poland)",
        ],
    },
    "Ib": {
        "name": "Wilderness Area",
        "description": (
            "Large unmodified or slightly modified area, retaining its "
            "natural character and influence, without permanent or "
            "significant human habitation."
        ),
        "commodity_production_allowed": False,
        "commodity_production_note": "Absolutely prohibited",
        "risk_score": 100,
        "risk_level": "critical",
        "default_buffer_km": Decimal("10"),
        "examples": [
            "Wrangell-St. Elias Wilderness (USA)",
            "Tarkine Wilderness (Australia)",
        ],
    },
    "II": {
        "name": "National Park",
        "description": (
            "Large natural or near-natural area set aside to protect "
            "large-scale ecological processes, along with the complement "
            "of species and ecosystems characteristic of the area."
        ),
        "commodity_production_allowed": False,
        "commodity_production_note": "Prohibited (except limited traditional use)",
        "risk_score": 95,
        "risk_level": "critical",
        "default_buffer_km": Decimal("10"),
        "examples": [
            "Virunga National Park (DRC)",
            "Tanjung Puting National Park (Indonesia)",
            "Manu National Park (Peru)",
        ],
    },
    "III": {
        "name": "Natural Monument or Feature",
        "description": (
            "Area set aside to protect a specific natural monument, "
            "which can be a landform, sea mount, submarine cavern, "
            "geological feature or even a living feature."
        ),
        "commodity_production_allowed": False,
        "commodity_production_note": "Prohibited",
        "risk_score": 85,
        "risk_level": "high",
        "default_buffer_km": Decimal("5"),
        "examples": [
            "Devil's Tower National Monument (USA)",
            "Iguazu Falls (Argentina/Brazil)",
        ],
    },
    "IV": {
        "name": "Habitat/Species Management Area",
        "description": (
            "Area aimed at protecting particular species or habitats "
            "where management reflects this priority. Many areas will "
            "need regular, active interventions to address the "
            "requirements of particular species or habitats."
        ),
        "commodity_production_allowed": False,
        "commodity_production_note": "Generally prohibited; limited exceptions",
        "risk_score": 80,
        "risk_level": "high",
        "default_buffer_km": Decimal("5"),
        "examples": [
            "Kinabatangan Wildlife Sanctuary (Malaysia)",
            "Dja Faunal Reserve (Cameroon)",
        ],
    },
    "V": {
        "name": "Protected Landscape/Seascape",
        "description": (
            "Area where the interaction of people and nature over time "
            "has produced an area of distinct character with significant "
            "ecological, biological, cultural and scenic value."
        ),
        "commodity_production_allowed": True,
        "commodity_production_note": "Conditionally allowed (sustainable practices)",
        "risk_score": 50,
        "risk_level": "medium",
        "default_buffer_km": Decimal("2"),
        "examples": [
            "Lake District National Park (UK)",
            "Cinque Terre National Park (Italy)",
        ],
    },
    "VI": {
        "name": "Protected Area with Sustainable Use of Natural Resources",
        "description": (
            "Area that conserves ecosystems and habitats, together with "
            "associated cultural values and traditional natural resource "
            "management systems, generally large, with most of the area "
            "in a natural condition."
        ),
        "commodity_production_allowed": True,
        "commodity_production_note": "Allowed under management plan",
        "risk_score": 25,
        "risk_level": "low",
        "default_buffer_km": Decimal("1"),
        "examples": [
            "Mamirauá Sustainable Development Reserve (Brazil)",
            "Tsavo West National Reserve (Kenya)",
        ],
    },
    "NR": {
        "name": "Not Reported",
        "description": (
            "IUCN management category not assigned or not reported to "
            "WDPA. Treated as HIGH risk by default due to uncertainty."
        ),
        "commodity_production_allowed": None,
        "commodity_production_note": "Uncertain - treated as HIGH by default",
        "risk_score": 75,
        "risk_level": "high",
        "default_buffer_km": Decimal("5"),
        "examples": [],
    },
}


def get_iucn_risk_score(category: str) -> int:
    """Get the base risk score for an IUCN category.

    Args:
        category: IUCN category string (Ia, Ib, II, III, IV, V, VI, NR).

    Returns:
        Integer risk score (0-100).

    Raises:
        ValueError: If category is not recognized.

    Example:
        >>> get_iucn_risk_score("Ia")
        100
        >>> get_iucn_risk_score("VI")
        25
    """
    if category not in IUCN_CATEGORY_SCORES:
        raise ValueError(
            f"Unknown IUCN category: {category}. "
            f"Must be one of {sorted(IUCN_CATEGORY_SCORES.keys())}"
        )
    return IUCN_CATEGORY_SCORES[category]


def get_iucn_definition(category: str) -> Dict[str, Any]:
    """Get the full definition for an IUCN category.

    Args:
        category: IUCN category string.

    Returns:
        Dictionary with name, description, risk_score, and other metadata.

    Raises:
        ValueError: If category is not recognized.
    """
    if category not in IUCN_CATEGORY_DEFINITIONS:
        raise ValueError(
            f"Unknown IUCN category: {category}. "
            f"Must be one of {sorted(IUCN_CATEGORY_DEFINITIONS.keys())}"
        )
    return IUCN_CATEGORY_DEFINITIONS[category]


def get_default_buffer_km(category: str) -> Decimal:
    """Get the default buffer zone radius for an IUCN category.

    Args:
        category: IUCN category string.

    Returns:
        Default buffer radius in kilometers.
    """
    defn = get_iucn_definition(category)
    return defn["default_buffer_km"]
