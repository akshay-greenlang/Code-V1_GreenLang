# -*- coding: utf-8 -*-
"""
Forest Classification Thresholds - AGENT-EUDR-003

Provides NDVI and EVI classification thresholds per biome type for forest
cover assessment. Each biome has distinct spectral characteristics that
require calibrated thresholds for accurate classification of dense forest,
forest, shrubland, sparse vegetation, and non-vegetated land covers.

Biome-specific thresholds are derived from peer-reviewed literature:
    - Huete et al. (2002) - EVI characterization of tropical biomes
    - Myneni et al. (2007) - NDVI global vegetation monitoring
    - Hansen et al. (2013) - Global forest cover change datasets
    - IPCC AFOLU Volume 4 - Biome classification guidelines

EUDR Relevance:
    Article 2(1) defines "deforestation" as conversion of forest to
    agricultural use. Accurate forest classification is essential for
    establishing pre-cutoff baselines and detecting post-cutoff change.

Data source: Calibrated from MODIS/Sentinel-2 validation studies.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# NDVI thresholds per biome type
# ---------------------------------------------------------------------------
#
# Each entry maps a biome name to a tuple of NDVI classification thresholds:
#     (dense_forest, forest, shrubland, sparse)
#
# Classification rules:
#     NDVI >= dense_forest    -> "dense_forest"
#     NDVI >= forest          -> "forest"
#     NDVI >= shrubland       -> "shrubland"
#     NDVI >= sparse          -> "sparse_vegetation"
#     NDVI <  sparse          -> "non_vegetated"

BIOME_NDVI_THRESHOLDS: Dict[str, Tuple[float, float, float, float]] = {
    # Tropical biomes -- high baseline NDVI, dense canopy year-round
    "tropical_rainforest": (0.65, 0.45, 0.25, 0.05),
    "tropical_dry_forest": (0.55, 0.35, 0.20, 0.05),
    "tropical_moist_forest": (0.60, 0.42, 0.22, 0.05),

    # Temperate biomes -- seasonal NDVI variation, deciduous component
    "temperate_forest": (0.60, 0.40, 0.20, 0.05),
    "temperate_rainforest": (0.62, 0.43, 0.22, 0.05),
    "temperate_deciduous": (0.58, 0.38, 0.18, 0.05),

    # Boreal biomes -- lower peak NDVI, coniferous dominance
    "boreal_forest": (0.50, 0.30, 0.15, 0.05),

    # Coastal and wetland biomes
    "mangrove": (0.50, 0.30, 0.15, 0.00),
    "peat_swamp_forest": (0.55, 0.35, 0.18, 0.02),

    # Savanna and cerrado biomes -- grass/tree mosaic
    "cerrado_savanna": (0.45, 0.25, 0.15, 0.05),
    "tropical_savanna": (0.42, 0.25, 0.15, 0.05),
    "woodland_savanna": (0.48, 0.28, 0.16, 0.05),

    # Montane biomes -- altitude-reduced NDVI
    "montane_cloud_forest": (0.55, 0.38, 0.20, 0.05),
    "montane_dry_forest": (0.48, 0.32, 0.18, 0.05),

    # Arid and semi-arid biomes
    "dry_woodland": (0.40, 0.22, 0.12, 0.03),
    "thorn_forest": (0.38, 0.20, 0.10, 0.03),
}

# ---------------------------------------------------------------------------
# EVI thresholds per biome type
# ---------------------------------------------------------------------------
#
# EVI is less sensitive to atmospheric effects and soil background than
# NDVI, making it preferred for dense tropical canopies. Format matches
# NDVI thresholds: (dense_forest, forest, shrubland, sparse).

BIOME_EVI_THRESHOLDS: Dict[str, Tuple[float, float, float, float]] = {
    "tropical_rainforest": (0.45, 0.30, 0.15, 0.05),
    "tropical_dry_forest": (0.38, 0.25, 0.12, 0.04),
    "tropical_moist_forest": (0.42, 0.28, 0.14, 0.05),
    "temperate_forest": (0.40, 0.27, 0.13, 0.04),
    "temperate_rainforest": (0.43, 0.29, 0.14, 0.04),
    "temperate_deciduous": (0.38, 0.25, 0.12, 0.04),
    "boreal_forest": (0.32, 0.20, 0.10, 0.04),
    "mangrove": (0.35, 0.22, 0.10, 0.02),
    "peat_swamp_forest": (0.38, 0.24, 0.12, 0.03),
    "cerrado_savanna": (0.30, 0.18, 0.10, 0.04),
    "tropical_savanna": (0.28, 0.17, 0.10, 0.04),
    "woodland_savanna": (0.32, 0.20, 0.11, 0.04),
    "montane_cloud_forest": (0.38, 0.25, 0.13, 0.04),
    "montane_dry_forest": (0.33, 0.21, 0.11, 0.04),
    "dry_woodland": (0.26, 0.15, 0.08, 0.03),
    "thorn_forest": (0.24, 0.14, 0.07, 0.03),
}


# ---------------------------------------------------------------------------
# Commodity-to-biome mapping per country
# ---------------------------------------------------------------------------
#
# Maps EUDR-regulated commodities to their typical biome types in each
# producing country. Used for automatic biome selection when the biome
# is not explicitly provided by the operator.
#
# Key: commodity (lowercase)
# Value: dict mapping ISO 3166-1 alpha-2 country code to biome name

COMMODITY_BIOME_MAP: Dict[str, Dict[str, str]] = {
    "palm_oil": {
        "ID": "tropical_rainforest",     # Indonesia - Sumatra, Kalimantan
        "MY": "tropical_rainforest",     # Malaysia - Sabah, Sarawak
        "TH": "tropical_moist_forest",   # Thailand
        "CO": "tropical_moist_forest",   # Colombia
        "NG": "tropical_moist_forest",   # Nigeria
        "GH": "tropical_moist_forest",   # Ghana
        "PG": "tropical_rainforest",     # Papua New Guinea
        "CM": "tropical_moist_forest",   # Cameroon
        "CI": "tropical_moist_forest",   # Ivory Coast
        "HN": "tropical_moist_forest",   # Honduras
        "GT": "tropical_moist_forest",   # Guatemala
        "CR": "tropical_moist_forest",   # Costa Rica
    },
    "soya": {
        "BR": "cerrado_savanna",         # Brazil - Cerrado, Matopiba
        "AR": "tropical_savanna",        # Argentina - Chaco, Pampa
        "PY": "cerrado_savanna",         # Paraguay - Chaco
        "BO": "tropical_dry_forest",     # Bolivia - lowlands
        "UY": "tropical_savanna",        # Uruguay
        "US": "temperate_deciduous",     # United States
    },
    "cattle": {
        "BR": "cerrado_savanna",         # Brazil - Cerrado, Amazon arc
        "AR": "tropical_savanna",        # Argentina - Chaco
        "PY": "cerrado_savanna",         # Paraguay - Chaco
        "CO": "tropical_savanna",        # Colombia - Llanos
        "BO": "tropical_dry_forest",     # Bolivia
        "UY": "tropical_savanna",        # Uruguay
    },
    "cocoa": {
        "CI": "tropical_moist_forest",   # Ivory Coast
        "GH": "tropical_moist_forest",   # Ghana
        "CM": "tropical_moist_forest",   # Cameroon
        "NG": "tropical_moist_forest",   # Nigeria
        "EC": "tropical_moist_forest",   # Ecuador
        "BR": "tropical_rainforest",     # Brazil - Bahia, Para
        "ID": "tropical_rainforest",     # Indonesia
        "PE": "tropical_moist_forest",   # Peru
        "CD": "tropical_rainforest",     # DR Congo
        "CG": "tropical_rainforest",     # Republic of Congo
    },
    "coffee": {
        "BR": "tropical_moist_forest",   # Brazil - Minas Gerais
        "VN": "tropical_moist_forest",   # Vietnam - Central Highlands
        "CO": "montane_cloud_forest",    # Colombia - Eje Cafetero
        "ID": "tropical_rainforest",     # Indonesia - Sumatra
        "ET": "montane_cloud_forest",    # Ethiopia - highland forests
        "HN": "montane_cloud_forest",    # Honduras
        "PE": "montane_cloud_forest",    # Peru
        "GT": "montane_cloud_forest",    # Guatemala
        "UG": "tropical_moist_forest",   # Uganda
        "MX": "montane_dry_forest",      # Mexico - Chiapas
        "KE": "montane_cloud_forest",    # Kenya
        "TZ": "montane_cloud_forest",    # Tanzania
        "IN": "tropical_moist_forest",   # India
        "RW": "montane_cloud_forest",    # Rwanda
    },
    "rubber": {
        "TH": "tropical_moist_forest",   # Thailand
        "ID": "tropical_rainforest",     # Indonesia
        "VN": "tropical_moist_forest",   # Vietnam
        "MY": "tropical_rainforest",     # Malaysia
        "IN": "tropical_moist_forest",   # India - Kerala
        "CI": "tropical_moist_forest",   # Ivory Coast
        "CM": "tropical_moist_forest",   # Cameroon
        "MM": "tropical_moist_forest",   # Myanmar
        "KH": "tropical_moist_forest",   # Cambodia
        "LA": "tropical_moist_forest",   # Laos
        "LR": "tropical_moist_forest",   # Liberia
        "GH": "tropical_moist_forest",   # Ghana
    },
    "wood": {
        "BR": "tropical_rainforest",     # Brazil - Amazon
        "ID": "tropical_rainforest",     # Indonesia
        "MY": "tropical_rainforest",     # Malaysia
        "CD": "tropical_rainforest",     # DR Congo
        "CG": "tropical_rainforest",     # Republic of Congo
        "GA": "tropical_rainforest",     # Gabon
        "CM": "tropical_moist_forest",   # Cameroon
        "PE": "tropical_rainforest",     # Peru
        "BO": "tropical_dry_forest",     # Bolivia
        "PG": "tropical_rainforest",     # Papua New Guinea
        "MM": "tropical_moist_forest",   # Myanmar
        "GH": "tropical_moist_forest",   # Ghana
        "CI": "tropical_moist_forest",   # Ivory Coast
        "VN": "tropical_moist_forest",   # Vietnam
        "GQ": "tropical_rainforest",     # Equatorial Guinea
    },
}


# ---------------------------------------------------------------------------
# Classification levels
# ---------------------------------------------------------------------------

CLASSIFICATION_LEVELS: List[str] = [
    "dense_forest",
    "forest",
    "shrubland",
    "sparse_vegetation",
    "non_vegetated",
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_forest_threshold(
    biome: str,
    level: str = "forest",
    index_type: str = "ndvi",
) -> Optional[float]:
    """Get the spectral index threshold for a given biome and classification level.

    Args:
        biome: Biome name (e.g., 'tropical_rainforest', 'cerrado_savanna').
        level: Classification level. One of 'dense_forest', 'forest',
            'shrubland', or 'sparse'. Defaults to 'forest'.
        index_type: Spectral index type, 'ndvi' or 'evi'. Defaults to 'ndvi'.

    Returns:
        Threshold value for the specified biome and level, or None if
        the biome or level is not recognized.
    """
    level_map = {
        "dense_forest": 0,
        "forest": 1,
        "shrubland": 2,
        "sparse": 3,
    }

    idx = level_map.get(level)
    if idx is None:
        return None

    if index_type.lower() == "evi":
        thresholds = BIOME_EVI_THRESHOLDS.get(biome)
    else:
        thresholds = BIOME_NDVI_THRESHOLDS.get(biome)

    if thresholds is None:
        return None

    return thresholds[idx]


def classify_ndvi(
    ndvi_value: float,
    biome: str = "tropical_rainforest",
) -> str:
    """Classify a pixel based on its NDVI value and the biome context.

    Uses the biome-specific NDVI thresholds to assign the pixel to one
    of five classification levels. Falls back to tropical_rainforest
    thresholds if the biome is not recognized.

    Args:
        ndvi_value: NDVI value in range [-1.0, 1.0].
        biome: Biome name for threshold selection. Defaults to
            'tropical_rainforest'.

    Returns:
        Classification label: one of 'dense_forest', 'forest',
        'shrubland', 'sparse_vegetation', or 'non_vegetated'.
    """
    thresholds = BIOME_NDVI_THRESHOLDS.get(biome)
    if thresholds is None:
        # Fallback to tropical rainforest (most conservative thresholds)
        thresholds = BIOME_NDVI_THRESHOLDS["tropical_rainforest"]

    dense_forest, forest, shrubland, sparse = thresholds

    if ndvi_value >= dense_forest:
        return "dense_forest"
    if ndvi_value >= forest:
        return "forest"
    if ndvi_value >= shrubland:
        return "shrubland"
    if ndvi_value >= sparse:
        return "sparse_vegetation"
    return "non_vegetated"


def classify_evi(
    evi_value: float,
    biome: str = "tropical_rainforest",
) -> str:
    """Classify a pixel based on its EVI value and the biome context.

    Uses the biome-specific EVI thresholds to assign the pixel to one
    of five classification levels. Falls back to tropical_rainforest
    thresholds if the biome is not recognized.

    Args:
        evi_value: EVI value in range [-1.0, 1.0].
        biome: Biome name for threshold selection. Defaults to
            'tropical_rainforest'.

    Returns:
        Classification label: one of 'dense_forest', 'forest',
        'shrubland', 'sparse_vegetation', or 'non_vegetated'.
    """
    thresholds = BIOME_EVI_THRESHOLDS.get(biome)
    if thresholds is None:
        thresholds = BIOME_EVI_THRESHOLDS["tropical_rainforest"]

    dense_forest, forest, shrubland, sparse = thresholds

    if evi_value >= dense_forest:
        return "dense_forest"
    if evi_value >= forest:
        return "forest"
    if evi_value >= shrubland:
        return "shrubland"
    if evi_value >= sparse:
        return "sparse_vegetation"
    return "non_vegetated"


def get_biome_for_commodity(
    commodity: str,
    country_code: str,
) -> Optional[str]:
    """Determine the typical biome type for a commodity in a country.

    Uses the COMMODITY_BIOME_MAP to find the most likely biome where
    the specified commodity is produced in the given country.

    Args:
        commodity: EUDR commodity identifier (lowercase). One of:
            'palm_oil', 'soya', 'cattle', 'cocoa', 'coffee',
            'rubber', 'wood'.
        country_code: ISO 3166-1 alpha-2 country code (uppercase).

    Returns:
        Biome name string (e.g., 'tropical_rainforest'), or None if
        the commodity/country combination is not in the reference data.
    """
    commodity_map = COMMODITY_BIOME_MAP.get(commodity.lower())
    if commodity_map is None:
        return None
    return commodity_map.get(country_code.upper())


def get_all_biomes() -> List[str]:
    """Return a sorted list of all recognized biome names.

    Returns:
        Sorted list of biome name strings.
    """
    return sorted(BIOME_NDVI_THRESHOLDS.keys())


def is_forest_cover(
    ndvi_value: float,
    biome: str = "tropical_rainforest",
) -> bool:
    """Determine if an NDVI value indicates forest cover for the given biome.

    A pixel is considered forest if its NDVI exceeds the 'forest'
    threshold for the specified biome. This is a convenience wrapper
    around ``classify_ndvi()`` for boolean forest/non-forest decisions.

    Args:
        ndvi_value: NDVI value in range [-1.0, 1.0].
        biome: Biome name for threshold selection.

    Returns:
        True if the NDVI indicates forest cover (forest or dense_forest).
    """
    classification = classify_ndvi(ndvi_value, biome)
    return classification in ("forest", "dense_forest")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BIOME_NDVI_THRESHOLDS",
    "BIOME_EVI_THRESHOLDS",
    "COMMODITY_BIOME_MAP",
    "CLASSIFICATION_LEVELS",
    "get_forest_threshold",
    "classify_ndvi",
    "classify_evi",
    "get_biome_for_commodity",
    "get_all_biomes",
    "is_forest_cover",
]
