# -*- coding: utf-8 -*-
"""
Forest Definitions - AGENT-EUDR-004: Forest Cover Analysis Agent

Codifies the FAO and EUDR forest definitions as Python data structures for
deterministic (zero-hallucination) forest classification. These definitions
are the legal and scientific basis for all forest/non-forest determinations
in the Forest Cover Analysis Agent.

Definitions encoded:
    - FAO Forest Definition (FAO FRA 2020, EUDR Art. 2(4))
    - EUDR Forest Exclusions (plantation vs. natural forest distinctions)
    - Forest Type Classification Criteria (spectral, structural, phenological)
    - Deforestation and Degradation Definitions (EUDR, FAO, degradation)

Data sources:
    - FAO Global Forest Resources Assessment 2020 (FRA 2020)
    - Regulation (EU) 2023/1115 (EUDR), Articles 2, 3, and Recital 32
    - EU Observatory Technical Guidance (2024)
    - IPCC AFOLU Volume 4, Chapter 4 - Forest Land
    - Hansen et al. (2013) - Global tree cover definitions

EUDR Relevance:
    Accurate forest definition is the foundation of EUDR compliance.
    Article 2(4) references the FAO definition but adds critical
    exclusions for monoculture plantations. Misclassifying a plantation
    as forest (or vice versa) directly impacts the deforestation-free
    determination required by Article 3.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# FAO Forest Definition
# ---------------------------------------------------------------------------
#
# The Food and Agriculture Organization of the United Nations defines
# "forest" as land spanning more than 0.5 hectares with trees higher
# than 5 metres and a canopy cover of more than 10 per cent, or trees
# able to reach these thresholds in situ.
#
# EUDR Art. 2(4) adopts this definition with additional exclusions.

FAO_FOREST_DEFINITION: Dict[str, Any] = {
    "min_area_ha": 0.5,
    "min_canopy_cover_pct": 10.0,
    "min_tree_height_m": 5.0,
    "excludes": [
        "agricultural_tree_plantation_monoculture",
    ],
    "includes": [
        "temporarily_unstocked",
    ],
    "source": "FAO FRA 2020 / EUDR Art. 2(4)",
    "reference_url": (
        "https://www.fao.org/3/I8661EN/i8661en.pdf"
    ),
    "eudr_article": "Article 2(4)",
    "notes": (
        "Land predominantly under agricultural or urban land use is "
        "excluded. Temporarily unstocked forest land (e.g., clear-cut "
        "areas expected to regenerate) is included. Trees must be able "
        "to reach 5 m height and 10% canopy cover in situ."
    ),
}


# ---------------------------------------------------------------------------
# EUDR Forest Exclusions
# ---------------------------------------------------------------------------
#
# The EUDR modifies the FAO forest definition with commodity-specific
# exclusions that determine whether a land parcel with tree cover
# qualifies as "forest" for deforestation assessment purposes.

EUDR_FOREST_EXCLUSIONS: List[Dict[str, str]] = [
    {
        "commodity": "OIL_PALM",
        "exclusion_rule": (
            "Oil palm monoculture plantations are NOT classified as "
            "forest under the EUDR, regardless of canopy cover, height, "
            "or area. Conversion of natural forest to oil palm plantation "
            "constitutes deforestation."
        ),
        "article_reference": "EUDR Art. 2(4), Recital 32",
        "description": (
            "Oil palm plantations (Elaeis guineensis) grown as "
            "monoculture do not meet the EUDR forest definition. This "
            "prevents operators from claiming that replacing natural "
            "forest with palm plantation is not deforestation."
        ),
    },
    {
        "commodity": "RUBBER",
        "exclusion_rule": (
            "Rubber monoculture plantations (Hevea brasiliensis) are "
            "NOT classified as forest under the EUDR. Conversion of "
            "natural forest to rubber monoculture constitutes "
            "deforestation."
        ),
        "article_reference": "EUDR Art. 2(4), Recital 32",
        "description": (
            "Rubber tree plantations in monoculture configuration, "
            "despite meeting FAO canopy cover and height thresholds, "
            "do not qualify as forest for EUDR purposes."
        ),
    },
    {
        "commodity": "COCOA",
        "exclusion_rule": (
            "Cocoa grown under full-sun monoculture is NOT forest. "
            "However, shade-grown cocoa agroforestry systems with "
            "more than 10% native canopy cover ARE classified as "
            "forest per EUDR."
        ),
        "article_reference": "EUDR Art. 2(4), EU Observatory Guidance",
        "description": (
            "The distinction between sun-grown and shade-grown cocoa "
            "is critical. Shade-grown systems that maintain native "
            "forest canopy above the FAO 10% threshold retain forest "
            "classification and are considered deforestation-free."
        ),
    },
    {
        "commodity": "COFFEE",
        "exclusion_rule": (
            "Coffee monoculture plantations are NOT forest. "
            "Shade-grown coffee under native canopy with more than "
            "10% native tree cover IS classified as forest."
        ),
        "article_reference": "EUDR Art. 2(4), EU Observatory Guidance",
        "description": (
            "Similar to cocoa, the presence or absence of native "
            "canopy determines forest status. Traditional shade coffee "
            "systems (e.g., Ethiopian forest coffee) retain forest "
            "classification."
        ),
    },
    {
        "commodity": "WOOD",
        "exclusion_rule": (
            "Managed natural forests and semi-natural forests with "
            "selective logging ARE classified as forest. Intensively "
            "managed plantation forests (e.g., Eucalyptus monoculture) "
            "are NOT classified as forest per EUDR."
        ),
        "article_reference": "EUDR Art. 2(4)",
        "description": (
            "The EUDR distinguishes between natural/semi-natural "
            "managed forests (which are forest) and industrial tree "
            "plantations (which are not). Selective logging in natural "
            "forest does not change forest classification unless "
            "canopy cover falls permanently below 10%."
        ),
    },
    {
        "commodity": "SOYA",
        "exclusion_rule": (
            "Soya is grown on agricultural land and never qualifies as "
            "forest. The relevant assessment is whether the agricultural "
            "land was converted from forest after the cutoff date."
        ),
        "article_reference": "EUDR Art. 2(1), Art. 3",
        "description": (
            "For soya, the deforestation assessment focuses on the "
            "prior land use history: was the production plot forested "
            "as of December 31, 2020, and was it subsequently "
            "converted to soya production?"
        ),
    },
    {
        "commodity": "CATTLE",
        "exclusion_rule": (
            "Cattle pasture is agricultural land and does not qualify as "
            "forest. The assessment focuses on whether pastureland was "
            "converted from forest after the cutoff date."
        ),
        "article_reference": "EUDR Art. 2(1), Art. 3",
        "description": (
            "For cattle, deforestation assessment examines whether "
            "the grazing land or feedlot area was forested as of "
            "December 31, 2020, and subsequently cleared for cattle "
            "production."
        ),
    },
]


# ---------------------------------------------------------------------------
# Forest Type Classification Criteria
# ---------------------------------------------------------------------------
#
# Defines spectral, structural, and phenological criteria for each
# forest type classification used by the ForestTypeClassifier engine.
#
# Each entry includes:
#   min_canopy_pct         - Minimum canopy cover percentage
#   min_height_m           - Minimum tree height in metres
#   ndvi_range             - (min, max) typical NDVI range
#   typical_agb_range      - (min, max) AGB in Mg/ha
#   phenological_pattern   - Leaf cycle pattern
#   is_forest_per_fao      - Whether it meets FAO forest definition
#   is_forest_per_eudr     - Whether it meets EUDR forest definition
#   spectral_signature     - Band reflectance ranges for 6 bands

FOREST_TYPE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "dense_primary_tropical": {
        "min_canopy_pct": 70.0,
        "min_height_m": 25.0,
        "ndvi_range": (0.65, 0.95),
        "typical_agb_range": (200.0, 450.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.01, 0.04),
            "green": (0.03, 0.08),
            "red": (0.01, 0.05),
            "nir": (0.25, 0.50),
            "swir1": (0.08, 0.18),
            "swir2": (0.03, 0.10),
        },
    },
    "open_tropical_forest": {
        "min_canopy_pct": 40.0,
        "min_height_m": 10.0,
        "ndvi_range": (0.45, 0.70),
        "typical_agb_range": (80.0, 200.0),
        "phenological_pattern": "semi-deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.06),
            "green": (0.04, 0.10),
            "red": (0.03, 0.08),
            "nir": (0.20, 0.42),
            "swir1": (0.10, 0.22),
            "swir2": (0.05, 0.14),
        },
    },
    "dry_deciduous_forest": {
        "min_canopy_pct": 30.0,
        "min_height_m": 8.0,
        "ndvi_range": (0.30, 0.65),
        "typical_agb_range": (50.0, 150.0),
        "phenological_pattern": "deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.03, 0.08),
            "green": (0.05, 0.12),
            "red": (0.04, 0.12),
            "nir": (0.18, 0.38),
            "swir1": (0.12, 0.25),
            "swir2": (0.06, 0.16),
        },
    },
    "temperate_broadleaf_forest": {
        "min_canopy_pct": 40.0,
        "min_height_m": 12.0,
        "ndvi_range": (0.40, 0.85),
        "typical_agb_range": (100.0, 300.0),
        "phenological_pattern": "deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.06),
            "green": (0.03, 0.10),
            "red": (0.02, 0.08),
            "nir": (0.22, 0.48),
            "swir1": (0.08, 0.20),
            "swir2": (0.04, 0.12),
        },
    },
    "coniferous_forest": {
        "min_canopy_pct": 30.0,
        "min_height_m": 10.0,
        "ndvi_range": (0.35, 0.80),
        "typical_agb_range": (80.0, 350.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.01, 0.04),
            "green": (0.02, 0.07),
            "red": (0.01, 0.05),
            "nir": (0.18, 0.40),
            "swir1": (0.06, 0.16),
            "swir2": (0.02, 0.08),
        },
    },
    "boreal_forest": {
        "min_canopy_pct": 15.0,
        "min_height_m": 5.0,
        "ndvi_range": (0.25, 0.65),
        "typical_agb_range": (30.0, 150.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.01, 0.05),
            "green": (0.02, 0.08),
            "red": (0.01, 0.06),
            "nir": (0.15, 0.35),
            "swir1": (0.05, 0.14),
            "swir2": (0.02, 0.08),
        },
    },
    "mangrove_forest": {
        "min_canopy_pct": 25.0,
        "min_height_m": 5.0,
        "ndvi_range": (0.30, 0.75),
        "typical_agb_range": (80.0, 300.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.06),
            "green": (0.03, 0.08),
            "red": (0.02, 0.06),
            "nir": (0.15, 0.38),
            "swir1": (0.10, 0.22),
            "swir2": (0.05, 0.14),
        },
    },
    "montane_cloud_forest": {
        "min_canopy_pct": 40.0,
        "min_height_m": 10.0,
        "ndvi_range": (0.40, 0.80),
        "typical_agb_range": (80.0, 250.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.06),
            "green": (0.03, 0.09),
            "red": (0.02, 0.07),
            "nir": (0.20, 0.42),
            "swir1": (0.08, 0.18),
            "swir2": (0.04, 0.10),
        },
    },
    "plantation_monoculture": {
        "min_canopy_pct": 40.0,
        "min_height_m": 8.0,
        "ndvi_range": (0.45, 0.85),
        "typical_agb_range": (40.0, 150.0),
        "phenological_pattern": "evergreen",
        "is_forest_per_fao": False,
        "is_forest_per_eudr": False,
        "spectral_signature": {
            "blue": (0.02, 0.05),
            "green": (0.04, 0.09),
            "red": (0.02, 0.06),
            "nir": (0.22, 0.45),
            "swir1": (0.10, 0.20),
            "swir2": (0.04, 0.12),
        },
    },
    "agroforestry_native_canopy": {
        "min_canopy_pct": 10.0,
        "min_height_m": 5.0,
        "ndvi_range": (0.35, 0.78),
        "typical_agb_range": (30.0, 120.0),
        "phenological_pattern": "semi-deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.07),
            "green": (0.04, 0.11),
            "red": (0.03, 0.09),
            "nir": (0.18, 0.40),
            "swir1": (0.10, 0.22),
            "swir2": (0.05, 0.14),
        },
    },
    "secondary_regrowth": {
        "min_canopy_pct": 10.0,
        "min_height_m": 5.0,
        "ndvi_range": (0.30, 0.72),
        "typical_agb_range": (25.0, 150.0),
        "phenological_pattern": "semi-deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.02, 0.07),
            "green": (0.04, 0.11),
            "red": (0.03, 0.10),
            "nir": (0.16, 0.38),
            "swir1": (0.10, 0.24),
            "swir2": (0.05, 0.15),
        },
    },
    "savanna_woodland": {
        "min_canopy_pct": 10.0,
        "min_height_m": 5.0,
        "ndvi_range": (0.25, 0.55),
        "typical_agb_range": (20.0, 80.0),
        "phenological_pattern": "deciduous",
        "is_forest_per_fao": True,
        "is_forest_per_eudr": True,
        "spectral_signature": {
            "blue": (0.04, 0.10),
            "green": (0.06, 0.14),
            "red": (0.05, 0.14),
            "nir": (0.15, 0.35),
            "swir1": (0.14, 0.28),
            "swir2": (0.08, 0.18),
        },
    },
}


# ---------------------------------------------------------------------------
# Deforestation and Degradation Definitions
# ---------------------------------------------------------------------------

DEFORESTATION_DEFINITIONS: Dict[str, str] = {
    "eudr": (
        "Conversion of forest to agricultural use, whether "
        "human-induced or not (EUDR Art. 2(3)). The cutoff date "
        "is December 31, 2020. Products placed on the EU market "
        "must be produced on land that was not subject to "
        "deforestation after this date."
    ),
    "fao": (
        "Conversion of forest to other land use or the long-term "
        "reduction of tree canopy cover below the minimum 10 per "
        "cent threshold (FAO FRA 2020). This definition does not "
        "require the conversion to be human-induced."
    ),
    "degradation": (
        "Structural changes within a forest that significantly "
        "reduce canopy cover or biomass stock but where the "
        "remaining tree cover stays above the 10 per cent FAO "
        "threshold. The EUDR does not directly regulate degradation "
        "but it is monitored as an early warning indicator for "
        "potential future deforestation."
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_forest_definition(framework: str) -> Optional[Dict[str, Any]]:
    """Return the forest definition for a given regulatory framework.

    Args:
        framework: Framework identifier. One of 'fao' or 'eudr'
            (case-insensitive).

    Returns:
        Dictionary with the forest definition parameters and metadata.
        For 'fao', returns FAO_FOREST_DEFINITION.
        For 'eudr', returns FAO_FOREST_DEFINITION augmented with
        EUDR-specific exclusions.
        Returns None if the framework is not recognized.

    Example:
        >>> defn = get_forest_definition("eudr")
        >>> defn["min_canopy_cover_pct"]
        10.0
    """
    fw = framework.lower().strip()

    if fw == "fao":
        return dict(FAO_FOREST_DEFINITION)

    if fw == "eudr":
        eudr_defn = dict(FAO_FOREST_DEFINITION)
        eudr_defn["exclusions"] = [
            {
                "commodity": ex["commodity"],
                "rule": ex["exclusion_rule"],
                "article": ex["article_reference"],
            }
            for ex in EUDR_FOREST_EXCLUSIONS
        ]
        eudr_defn["framework"] = "EUDR"
        eudr_defn["cutoff_date"] = "2020-12-31"
        return eudr_defn

    return None


def is_forest_per_fao(
    canopy_cover_pct: float,
    tree_height_m: float,
    area_ha: float,
) -> bool:
    """Determine if a land parcel qualifies as forest per FAO definition.

    Args:
        canopy_cover_pct: Canopy cover percentage (0-100).
        tree_height_m: Tree height at maturity in metres.
        area_ha: Area of the land parcel in hectares.

    Returns:
        True if all three FAO thresholds are met.
    """
    return (
        canopy_cover_pct >= FAO_FOREST_DEFINITION["min_canopy_cover_pct"]
        and tree_height_m >= FAO_FOREST_DEFINITION["min_tree_height_m"]
        and area_ha >= FAO_FOREST_DEFINITION["min_area_ha"]
    )


def is_forest_per_eudr(
    canopy_cover_pct: float,
    tree_height_m: float,
    area_ha: float,
    is_monoculture_plantation: bool = False,
) -> bool:
    """Determine if a land parcel qualifies as forest per EUDR definition.

    The EUDR adopts the FAO definition but excludes monoculture tree
    plantations (oil palm, rubber, eucalyptus, etc.).

    Args:
        canopy_cover_pct: Canopy cover percentage (0-100).
        tree_height_m: Tree height at maturity in metres.
        area_ha: Area of the land parcel in hectares.
        is_monoculture_plantation: Whether the land is a monoculture
            tree plantation. Defaults to False.

    Returns:
        True if the parcel meets FAO thresholds AND is not a
        monoculture plantation.
    """
    if is_monoculture_plantation:
        return False

    return is_forest_per_fao(canopy_cover_pct, tree_height_m, area_ha)


def get_forest_type_criteria(
    forest_type: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve the classification criteria for a specific forest type.

    Args:
        forest_type: Forest type identifier (e.g., 'dense_primary_tropical',
            'plantation_monoculture'). Must match a key in FOREST_TYPE_CRITERIA.

    Returns:
        Dictionary with spectral, structural, and regulatory criteria,
        or None if the forest type is not recognized.
    """
    return FOREST_TYPE_CRITERIA.get(forest_type)


def get_all_forest_types() -> List[str]:
    """Return a sorted list of all recognized forest type identifiers.

    Returns:
        Sorted list of forest type name strings.
    """
    return sorted(FOREST_TYPE_CRITERIA.keys())


def get_eudr_forest_types() -> List[str]:
    """Return forest types that qualify as forest under EUDR.

    Returns:
        List of forest type identifiers where is_forest_per_eudr is True.
    """
    return [
        ft for ft, criteria in FOREST_TYPE_CRITERIA.items()
        if criteria.get("is_forest_per_eudr", False)
    ]


def get_non_forest_types() -> List[str]:
    """Return forest types that do NOT qualify as forest under EUDR.

    Returns:
        List of forest type identifiers where is_forest_per_eudr is False.
    """
    return [
        ft for ft, criteria in FOREST_TYPE_CRITERIA.items()
        if not criteria.get("is_forest_per_eudr", True)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FAO_FOREST_DEFINITION",
    "EUDR_FOREST_EXCLUSIONS",
    "FOREST_TYPE_CRITERIA",
    "DEFORESTATION_DEFINITIONS",
    "get_forest_definition",
    "is_forest_per_fao",
    "is_forest_per_eudr",
    "get_forest_type_criteria",
    "get_all_forest_types",
    "get_eudr_forest_types",
    "get_non_forest_types",
]
