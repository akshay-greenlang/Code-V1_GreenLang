# -*- coding: utf-8 -*-
"""
IPCC Land Use Class Definitions and Parameters - AGENT-EUDR-005

Provides comprehensive land use classification parameters aligned with IPCC
2006 Guidelines Volume 4 (AFOLU) and FAO definitions. Each land use category
includes spectral thresholds, minimum area requirements, EUDR relevance
flags, and commodity mapping data.

EUDR Relevance:
    Article 2(1) defines "deforestation" as the conversion of forest to
    agricultural use or to other uses. Article 2(5) defines "forest
    degradation" as structural changes to forest land reducing canopy
    cover below thresholds. This module encodes all land use transitions
    that constitute deforestation or degradation under EUDR.

Data Sources:
    - IPCC 2006 Guidelines, Volume 4: Agriculture, Forestry and Other Land Use
    - FAO Global Forest Resources Assessment (FRA) 2020 definitions
    - EUDR Regulation (EU) 2023/1115, Articles 2(1), 2(4), 2(5)
    - Hansen et al. (2013) - Global forest cover change
    - Potapov et al. (2022) - Global oil palm and rubber plantation mapping

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent
Status: Production Ready
"""

from __future__ import annotations

import enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Land use category enumeration
# ---------------------------------------------------------------------------


class LandUseCategory(str, enum.Enum):
    """IPCC-aligned land use category enumeration.

    Ten categories covering all IPCC AFOLU land use classes plus
    EUDR-specific plantation types for commodity tracking.

    References:
        IPCC 2006 Guidelines, Volume 4, Chapter 3, Table 3.1
    """

    FOREST_LAND = "forest_land"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"
    OIL_PALM_PLANTATION = "oil_palm_plantation"
    RUBBER_PLANTATION = "rubber_plantation"
    PLANTATION_FOREST = "plantation_forest"
    WATER_BODY = "water_body"


# ---------------------------------------------------------------------------
# Land use class parameter definitions
# ---------------------------------------------------------------------------
#
# Each category maps to a dict with complete classification parameters:
#   - display_name: Human-readable label
#   - description: Full description of the land use class
#   - ipcc_code: IPCC AFOLU category code
#   - fao_code: FAO Land Cover Classification System code
#   - spectral_thresholds: Expected ranges for vegetation indices and bands
#   - minimum_area_ha: Minimum mapping unit area in hectares
#   - is_agricultural: Whether this class represents agricultural land use
#   - is_forest: Whether this class meets FAO forest definition
#   - eudr_relevant: Whether this class is relevant for EUDR assessment
#   - typical_ndvi_seasonal_amplitude: Typical seasonal NDVI range
#

LAND_USE_CLASSES: Dict[LandUseCategory, Dict[str, Any]] = {
    LandUseCategory.FOREST_LAND: {
        "display_name": "Forest Land",
        "description": (
            "Land spanning more than 0.5 hectares with trees higher than "
            "5 metres and a canopy cover of more than 10 percent, or trees "
            "able to reach these thresholds in situ. It does not include "
            "land that is predominantly under agricultural or urban land use."
        ),
        "ipcc_code": "FL",
        "fao_code": "1",
        "spectral_thresholds": {
            "ndvi_range": (0.40, 0.95),
            "evi_range": (0.25, 0.60),
            "ndmi_range": (0.10, 0.60),
            "nir_range": (0.20, 0.55),
            "swir_range": (0.05, 0.25),
        },
        "minimum_area_ha": 0.5,
        "is_agricultural": False,
        "is_forest": True,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.15,
    },
    LandUseCategory.CROPLAND: {
        "display_name": "Cropland",
        "description": (
            "Arable and tillage land, rice fields, and agroforestry systems "
            "where the vegetation structure falls below the thresholds used "
            "for the forest land category. Includes both annual and perennial "
            "crops, and fallow land within crop rotation cycles."
        ),
        "ipcc_code": "CL",
        "fao_code": "2",
        "spectral_thresholds": {
            "ndvi_range": (0.15, 0.80),
            "evi_range": (0.10, 0.50),
            "ndmi_range": (-0.10, 0.40),
            "nir_range": (0.15, 0.50),
            "swir_range": (0.10, 0.35),
        },
        "minimum_area_ha": 0.25,
        "is_agricultural": True,
        "is_forest": False,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.40,
    },
    LandUseCategory.GRASSLAND: {
        "display_name": "Grassland",
        "description": (
            "Rangelands and pasture land that is not considered cropland. "
            "Includes natural grasslands, savannas, and grassland systems "
            "managed for livestock grazing. Also includes shrublands where "
            "the canopy cover is below forest thresholds."
        ),
        "ipcc_code": "GL",
        "fao_code": "3",
        "spectral_thresholds": {
            "ndvi_range": (0.10, 0.65),
            "evi_range": (0.05, 0.40),
            "ndmi_range": (-0.15, 0.30),
            "nir_range": (0.15, 0.45),
            "swir_range": (0.10, 0.30),
        },
        "minimum_area_ha": 0.5,
        "is_agricultural": True,
        "is_forest": False,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.30,
    },
    LandUseCategory.WETLAND: {
        "display_name": "Wetland",
        "description": (
            "Land that is covered or saturated by water for all or part of "
            "the year. Includes peatlands, mangroves, marshes, and other "
            "wetland ecosystems. Peatland drainage for agriculture is "
            "particularly relevant for EUDR palm oil supply chains."
        ),
        "ipcc_code": "WL",
        "fao_code": "4",
        "spectral_thresholds": {
            "ndvi_range": (0.05, 0.70),
            "evi_range": (0.02, 0.45),
            "ndmi_range": (0.20, 0.80),
            "nir_range": (0.05, 0.35),
            "swir_range": (0.02, 0.15),
        },
        "minimum_area_ha": 0.25,
        "is_agricultural": False,
        "is_forest": False,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.20,
    },
    LandUseCategory.SETTLEMENT: {
        "display_name": "Settlement",
        "description": (
            "All developed land, including transportation infrastructure "
            "and human settlements of any size. Includes residential, "
            "commercial, industrial, and institutional areas, as well as "
            "roads, railways, and other built-up surfaces."
        ),
        "ipcc_code": "SL",
        "fao_code": "5",
        "spectral_thresholds": {
            "ndvi_range": (-0.10, 0.30),
            "evi_range": (-0.05, 0.20),
            "ndmi_range": (-0.40, 0.05),
            "nir_range": (0.10, 0.35),
            "swir_range": (0.15, 0.45),
        },
        "minimum_area_ha": 0.1,
        "is_agricultural": False,
        "is_forest": False,
        "eudr_relevant": False,
        "typical_ndvi_seasonal_amplitude": 0.05,
    },
    LandUseCategory.OTHER_LAND: {
        "display_name": "Other Land",
        "description": (
            "Bare soil, rock, ice, and all land areas that do not fall "
            "into any of the other five IPCC categories. Includes desert, "
            "bare rock, sand dunes, and degraded land not classified "
            "elsewhere."
        ),
        "ipcc_code": "OL",
        "fao_code": "6",
        "spectral_thresholds": {
            "ndvi_range": (-0.20, 0.15),
            "evi_range": (-0.10, 0.10),
            "ndmi_range": (-0.50, -0.10),
            "nir_range": (0.15, 0.45),
            "swir_range": (0.20, 0.55),
        },
        "minimum_area_ha": 1.0,
        "is_agricultural": False,
        "is_forest": False,
        "eudr_relevant": False,
        "typical_ndvi_seasonal_amplitude": 0.05,
    },
    LandUseCategory.OIL_PALM_PLANTATION: {
        "display_name": "Oil Palm Plantation",
        "description": (
            "Land under oil palm (Elaeis guineensis) cultivation, including "
            "both industrial-scale plantations and smallholder plots. "
            "Characterized by distinctive spectral signatures that differ "
            "from natural forest due to monoculture canopy structure and "
            "regular planting geometry."
        ),
        "ipcc_code": "CL-P",
        "fao_code": "2.1",
        "spectral_thresholds": {
            "ndvi_range": (0.35, 0.80),
            "evi_range": (0.20, 0.50),
            "ndmi_range": (0.05, 0.40),
            "nir_range": (0.20, 0.50),
            "swir_range": (0.08, 0.28),
        },
        "minimum_area_ha": 0.25,
        "is_agricultural": True,
        "is_forest": False,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.10,
    },
    LandUseCategory.RUBBER_PLANTATION: {
        "display_name": "Rubber Plantation",
        "description": (
            "Land under rubber tree (Hevea brasiliensis) cultivation. "
            "Rubber plantations show seasonal leaf-shedding patterns in "
            "tropical dry seasons, creating a distinctive phenological "
            "signature with periodic NDVI drops that distinguish them "
            "from evergreen natural forest."
        ),
        "ipcc_code": "CL-R",
        "fao_code": "2.2",
        "spectral_thresholds": {
            "ndvi_range": (0.25, 0.80),
            "evi_range": (0.15, 0.50),
            "ndmi_range": (0.00, 0.40),
            "nir_range": (0.18, 0.48),
            "swir_range": (0.08, 0.30),
        },
        "minimum_area_ha": 0.25,
        "is_agricultural": True,
        "is_forest": False,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.25,
    },
    LandUseCategory.PLANTATION_FOREST: {
        "display_name": "Plantation Forest",
        "description": (
            "Intensively managed forest stands established by planting or "
            "seeding, with one or two species at regular spacing. Includes "
            "pulp and timber plantations (eucalyptus, pine, teak, acacia). "
            "Per EUDR Article 2(4), conversion between plantation forest "
            "types is not classified as deforestation."
        ),
        "ipcc_code": "FL-P",
        "fao_code": "1.2",
        "spectral_thresholds": {
            "ndvi_range": (0.35, 0.85),
            "evi_range": (0.20, 0.55),
            "ndmi_range": (0.05, 0.45),
            "nir_range": (0.18, 0.50),
            "swir_range": (0.06, 0.25),
        },
        "minimum_area_ha": 0.5,
        "is_agricultural": False,
        "is_forest": True,
        "eudr_relevant": True,
        "typical_ndvi_seasonal_amplitude": 0.15,
    },
    LandUseCategory.WATER_BODY: {
        "display_name": "Water Body",
        "description": (
            "Areas permanently or seasonally covered by water, including "
            "rivers, lakes, reservoirs, and coastal waters. Low NIR "
            "reflectance and strongly negative NDVI distinguish water "
            "from all vegetated classes."
        ),
        "ipcc_code": "WB",
        "fao_code": "7",
        "spectral_thresholds": {
            "ndvi_range": (-0.50, 0.00),
            "evi_range": (-0.30, 0.00),
            "ndmi_range": (0.50, 1.00),
            "nir_range": (0.00, 0.10),
            "swir_range": (0.00, 0.08),
        },
        "minimum_area_ha": 0.1,
        "is_agricultural": False,
        "is_forest": False,
        "eudr_relevant": False,
        "typical_ndvi_seasonal_amplitude": 0.02,
    },
}


# ---------------------------------------------------------------------------
# EUDR deforestation transitions
# ---------------------------------------------------------------------------
#
# Set of (from_class, to_class) tuples that constitute deforestation
# under EUDR Article 2(1): "conversion of forest to agricultural use
# or to other non-forest uses."
#
# A transition is classified as deforestation when:
#   1. The source class is a forest type (FOREST_LAND or PLANTATION_FOREST)
#   2. The destination class is non-forest
#   3. The transition is NOT excluded by Article 2(4)

EUDR_DEFORESTATION_TRANSITIONS: FrozenSet[Tuple[LandUseCategory, LandUseCategory]] = frozenset({
    # Natural forest -> agricultural
    (LandUseCategory.FOREST_LAND, LandUseCategory.CROPLAND),
    (LandUseCategory.FOREST_LAND, LandUseCategory.GRASSLAND),
    (LandUseCategory.FOREST_LAND, LandUseCategory.OIL_PALM_PLANTATION),
    (LandUseCategory.FOREST_LAND, LandUseCategory.RUBBER_PLANTATION),
    # Natural forest -> non-agricultural
    (LandUseCategory.FOREST_LAND, LandUseCategory.SETTLEMENT),
    (LandUseCategory.FOREST_LAND, LandUseCategory.OTHER_LAND),
    (LandUseCategory.FOREST_LAND, LandUseCategory.WATER_BODY),
    # Plantation forest -> agricultural
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.CROPLAND),
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.GRASSLAND),
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.OIL_PALM_PLANTATION),
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.RUBBER_PLANTATION),
    # Plantation forest -> non-agricultural non-forest
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.SETTLEMENT),
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.OTHER_LAND),
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.WATER_BODY),
    # Natural forest -> wetland (drainage/conversion)
    (LandUseCategory.FOREST_LAND, LandUseCategory.WETLAND),
    # Plantation forest -> wetland (drainage/conversion)
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.WETLAND),
})


# ---------------------------------------------------------------------------
# EUDR degradation transitions
# ---------------------------------------------------------------------------
#
# Set of (from_class, to_class) tuples that constitute forest degradation
# under EUDR Article 2(5): "Structural changes to forest land in the form
# of conversion of primary forest or naturally regenerating forest to
# plantation forest."

EUDR_DEGRADATION_TRANSITIONS: FrozenSet[Tuple[LandUseCategory, LandUseCategory]] = frozenset({
    (LandUseCategory.FOREST_LAND, LandUseCategory.PLANTATION_FOREST),
})


# ---------------------------------------------------------------------------
# EUDR excluded transitions
# ---------------------------------------------------------------------------
#
# Transitions that are explicitly NOT considered deforestation per EUDR
# Article 2(4): "For the purposes of this Regulation, conversion of
# forest that is a plantation forest from one tree species to another
# shall not be considered deforestation."

EUDR_EXCLUDED_TRANSITIONS: FrozenSet[Tuple[LandUseCategory, LandUseCategory]] = frozenset({
    (LandUseCategory.PLANTATION_FOREST, LandUseCategory.PLANTATION_FOREST),
})


# ---------------------------------------------------------------------------
# Commodity -> expected land use category mapping
# ---------------------------------------------------------------------------
#
# Maps each EUDR-regulated commodity to the land use class that
# characterizes its production area. Used by CutoffDateVerifier and
# CroplandExpansionDetector for commodity-specific analysis.

COMMODITY_LAND_USE_MAP: Dict[str, LandUseCategory] = {
    "palm_oil": LandUseCategory.OIL_PALM_PLANTATION,
    "rubber": LandUseCategory.RUBBER_PLANTATION,
    "soya": LandUseCategory.CROPLAND,
    "cattle": LandUseCategory.GRASSLAND,
    "cocoa": LandUseCategory.CROPLAND,
    "coffee": LandUseCategory.CROPLAND,
    "wood": LandUseCategory.FOREST_LAND,
}


# ---------------------------------------------------------------------------
# Forest definition parameters (FAO / EUDR aligned)
# ---------------------------------------------------------------------------

FAO_FOREST_DEFINITION: Dict[str, Any] = {
    "minimum_area_ha": 0.5,
    "minimum_canopy_cover_pct": 10.0,
    "minimum_tree_height_m": 5.0,
    "minimum_width_m": 20.0,
    "description": (
        "FAO definition: Land spanning more than 0.5 hectares with trees "
        "higher than 5 metres and a canopy cover of more than 10 percent, "
        "or trees able to reach these thresholds in situ. Does not include "
        "land predominantly under agricultural or urban land use."
    ),
    "reference": "FAO Global Forest Resources Assessment 2020",
}


# ---------------------------------------------------------------------------
# EUDR cutoff date (Article 2(1))
# ---------------------------------------------------------------------------

EUDR_CUTOFF_DATE: str = "2020-12-31"
"""EUDR deforestation cutoff date in ISO 8601 format.

Per EUDR Article 2(1), commodities and products shall not be placed on
or exported from the Union market if they have been produced on land
that was subject to deforestation after 31 December 2020.
"""


# ---------------------------------------------------------------------------
# EUDR commodity list (Article 1(2), Annex I)
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "palm_oil",
    "soya",
    "cattle",
    "cocoa",
    "coffee",
    "rubber",
    "wood",
]
"""Seven EUDR-regulated commodities per Regulation (EU) 2023/1115 Annex I."""


# ---------------------------------------------------------------------------
# Transition reversibility classification
# ---------------------------------------------------------------------------

REVERSIBLE_TRANSITIONS: FrozenSet[Tuple[LandUseCategory, LandUseCategory]] = frozenset({
    # Agricultural abandonment -> natural regeneration
    (LandUseCategory.CROPLAND, LandUseCategory.GRASSLAND),
    (LandUseCategory.GRASSLAND, LandUseCategory.CROPLAND),
    # Plantation management cycles
    (LandUseCategory.OIL_PALM_PLANTATION, LandUseCategory.CROPLAND),
    (LandUseCategory.RUBBER_PLANTATION, LandUseCategory.CROPLAND),
    # Seasonal flooding
    (LandUseCategory.GRASSLAND, LandUseCategory.WETLAND),
    (LandUseCategory.WETLAND, LandUseCategory.GRASSLAND),
})


IRREVERSIBLE_TRANSITIONS: FrozenSet[Tuple[LandUseCategory, LandUseCategory]] = frozenset({
    # Urbanization is permanent
    (LandUseCategory.FOREST_LAND, LandUseCategory.SETTLEMENT),
    (LandUseCategory.CROPLAND, LandUseCategory.SETTLEMENT),
    (LandUseCategory.GRASSLAND, LandUseCategory.SETTLEMENT),
    (LandUseCategory.WETLAND, LandUseCategory.SETTLEMENT),
    # Reservoir creation is permanent
    (LandUseCategory.FOREST_LAND, LandUseCategory.WATER_BODY),
    (LandUseCategory.CROPLAND, LandUseCategory.WATER_BODY),
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_land_use_params(category: LandUseCategory) -> Dict[str, Any]:
    """Retrieve the full parameter dictionary for a land use category.

    Args:
        category: The LandUseCategory to look up.

    Returns:
        Dictionary containing all parameters for the land use class.

    Raises:
        KeyError: If the category is not found in LAND_USE_CLASSES.
    """
    if category not in LAND_USE_CLASSES:
        raise KeyError(
            f"Unknown land use category: {category}. "
            f"Valid categories: {[c.value for c in LandUseCategory]}"
        )
    return LAND_USE_CLASSES[category]


def is_forest_class(category: LandUseCategory) -> bool:
    """Determine whether a land use category qualifies as forest.

    A category qualifies as forest if it meets the FAO definition:
    >0.5 ha, >10% canopy cover, >5m tree height.

    Args:
        category: The LandUseCategory to check.

    Returns:
        True if the category is classified as forest.
    """
    params = LAND_USE_CLASSES.get(category)
    if params is None:
        return False
    return bool(params.get("is_forest", False))


def is_agricultural_class(category: LandUseCategory) -> bool:
    """Determine whether a land use category represents agricultural use.

    Agricultural classes include cropland, grassland (pastoral), and
    commodity-specific plantation types (oil palm, rubber).

    Args:
        category: The LandUseCategory to check.

    Returns:
        True if the category represents agricultural land use.
    """
    params = LAND_USE_CLASSES.get(category)
    if params is None:
        return False
    return bool(params.get("is_agricultural", False))


def is_eudr_relevant(category: LandUseCategory) -> bool:
    """Determine whether a land use category is relevant for EUDR assessment.

    EUDR-relevant categories are those that participate in transitions
    classified as deforestation, degradation, or commodity production.

    Args:
        category: The LandUseCategory to check.

    Returns:
        True if the category is relevant for EUDR compliance assessment.
    """
    params = LAND_USE_CLASSES.get(category)
    if params is None:
        return False
    return bool(params.get("eudr_relevant", False))


def is_deforestation_transition(
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> bool:
    """Determine whether a land use transition constitutes deforestation.

    A transition is deforestation per EUDR Article 2(1) if:
    - It is in the EUDR_DEFORESTATION_TRANSITIONS set
    - It is NOT in the EUDR_EXCLUDED_TRANSITIONS set

    Args:
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        True if the transition constitutes EUDR deforestation.
    """
    pair = (from_class, to_class)
    if pair in EUDR_EXCLUDED_TRANSITIONS:
        return False
    return pair in EUDR_DEFORESTATION_TRANSITIONS


def is_degradation_transition(
    from_class: LandUseCategory,
    to_class: LandUseCategory,
) -> bool:
    """Determine whether a land use transition constitutes degradation.

    A transition is degradation per EUDR Article 2(5) if it converts
    primary or naturally regenerating forest to plantation forest.

    Args:
        from_class: Source land use category.
        to_class: Destination land use category.

    Returns:
        True if the transition constitutes EUDR forest degradation.
    """
    return (from_class, to_class) in EUDR_DEGRADATION_TRANSITIONS


def get_commodity_land_use(commodity: str) -> Optional[LandUseCategory]:
    """Retrieve the expected land use category for an EUDR commodity.

    Args:
        commodity: EUDR commodity name (lowercase).

    Returns:
        The expected LandUseCategory, or None if the commodity is not
        in the EUDR commodity list.
    """
    return COMMODITY_LAND_USE_MAP.get(commodity.lower())


def get_spectral_thresholds(category: LandUseCategory) -> Dict[str, Tuple[float, float]]:
    """Retrieve spectral index thresholds for a land use category.

    Args:
        category: The LandUseCategory to look up.

    Returns:
        Dictionary mapping index names to (min, max) tuples.

    Raises:
        KeyError: If the category is not found in LAND_USE_CLASSES.
    """
    params = get_land_use_params(category)
    return dict(params.get("spectral_thresholds", {}))


def get_all_forest_classes() -> List[LandUseCategory]:
    """Return all land use categories classified as forest.

    Returns:
        List of LandUseCategory values where is_forest is True.
    """
    return [
        cat for cat, params in LAND_USE_CLASSES.items()
        if params.get("is_forest", False)
    ]


def get_all_agricultural_classes() -> List[LandUseCategory]:
    """Return all land use categories classified as agricultural.

    Returns:
        List of LandUseCategory values where is_agricultural is True.
    """
    return [
        cat for cat, params in LAND_USE_CLASSES.items()
        if params.get("is_agricultural", False)
    ]


def get_all_eudr_relevant_classes() -> List[LandUseCategory]:
    """Return all EUDR-relevant land use categories.

    Returns:
        List of LandUseCategory values where eudr_relevant is True.
    """
    return [
        cat for cat, params in LAND_USE_CLASSES.items()
        if params.get("eudr_relevant", False)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LandUseCategory",
    "LAND_USE_CLASSES",
    "EUDR_DEFORESTATION_TRANSITIONS",
    "EUDR_DEGRADATION_TRANSITIONS",
    "EUDR_EXCLUDED_TRANSITIONS",
    "COMMODITY_LAND_USE_MAP",
    "FAO_FOREST_DEFINITION",
    "EUDR_CUTOFF_DATE",
    "EUDR_COMMODITIES",
    "REVERSIBLE_TRANSITIONS",
    "IRREVERSIBLE_TRANSITIONS",
    "get_land_use_params",
    "is_forest_class",
    "is_agricultural_class",
    "is_eudr_relevant",
    "is_deforestation_transition",
    "is_degradation_transition",
    "get_commodity_land_use",
    "get_spectral_thresholds",
    "get_all_forest_classes",
    "get_all_agricultural_classes",
    "get_all_eudr_relevant_classes",
]
