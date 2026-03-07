# -*- coding: utf-8 -*-
"""
Biome Parameters - AGENT-EUDR-004: Forest Cover Analysis Agent

Provides biome-specific spectral, structural, and phenological parameters for
all 16 biome types used across the Forest Cover Analysis engines. Each biome
entry encodes NDVI thresholds, canopy regression coefficients, above-ground
biomass (AGB) ranges, typical canopy height ranges, degradation sensitivity,
and a 12-month phenological NDVI profile.

These parameters drive deterministic (zero-hallucination) calculations in:
    - CanopyDensityMapper: NDVI-to-canopy dimidiation model
    - ForestTypeClassifier: Biome-aware forest type classification
    - BiomassEstimator: AGB estimation from spectral indices
    - CanopyHeightModeler: Height estimation from AGB allometrics
    - DeforestationFreeVerifier: Biome-calibrated change thresholds
    - FragmentationAnalyzer: Biome-specific degradation detection

Biome selection sources:
    - Olson et al. (2001) - Terrestrial Ecoregions of the World (WWF)
    - IPCC AFOLU Volume 4 - Biome classification guidelines
    - Dinerstein et al. (2017) - Ecoregion-based approach to conservation
    - FAO FRA 2020 - Forest Resources Assessment biome categories

EUDR Relevance:
    Different biomes require different spectral thresholds for accurate
    forest/non-forest determination. Using a single global threshold
    produces unacceptable false positive/negative rates in savanna and
    agroforestry biomes. Biome-calibrated parameters enable Article 9
    due diligence accuracy requirements to be met across all EUDR
    commodity-producing regions.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Biome parameters dictionary
# ---------------------------------------------------------------------------
#
# Each entry maps a biome name to a dictionary of parameters:
#
#   ndvi_forest_threshold      - Minimum NDVI to consider pixel as forested
#   ndvi_soil_reference        - Bare soil NDVI for dimidiation model (NDVIs)
#   ndvi_vegetation_reference  - Full vegetation NDVI for dimidiation (NDVIv)
#   canopy_regression_a        - Linear NDVI-to-canopy slope coefficient
#   canopy_regression_b        - Linear NDVI-to-canopy intercept
#   typical_agb_range          - (min, max) above-ground biomass in Mg/ha
#   typical_height_range       - (min, max) canopy height in metres
#   degradation_threshold      - % canopy loss to classify as degradation
#   phenological_profile       - 12 monthly relative NDVI values (Jan-Dec)
#
# Dimidiation model for fractional vegetation cover (FVC):
#     FVC = ((NDVI - NDVIs) / (NDVIv - NDVIs))^2
#     canopy_density_pct = canopy_regression_a * NDVI + canopy_regression_b
#
# Phenological profile values represent relative NDVI fraction of annual
# maximum (1.0 = peak greenness). Used for seasonal normalization.

BIOME_PARAMETERS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # 1. Tropical Moist Broadleaf Forest
    # ------------------------------------------------------------------
    # Amazon, Congo Basin, Southeast Asia lowland dipterocarp forests.
    # Year-round high NDVI, minimal seasonal variation, very high AGB.
    "tropical_moist_broadleaf": {
        "ndvi_forest_threshold": 0.50,
        "ndvi_soil_reference": 0.05,
        "ndvi_vegetation_reference": 0.85,
        "canopy_regression_a": 140.0,
        "canopy_regression_b": -20.0,
        "typical_agb_range": (150.0, 450.0),
        "typical_height_range": (25.0, 55.0),
        "degradation_threshold": 15.0,
        "phenological_profile": [
            0.95, 0.96, 0.97, 0.98, 0.97, 0.95,
            0.93, 0.91, 0.90, 0.92, 0.93, 0.94,
        ],
    },

    # ------------------------------------------------------------------
    # 2. Tropical Dry Broadleaf Forest
    # ------------------------------------------------------------------
    # Cerrado-adjacent semi-deciduous forests, Chiquitano, Indochina.
    # Strong dry season signal, moderate AGB.
    "tropical_dry_broadleaf": {
        "ndvi_forest_threshold": 0.40,
        "ndvi_soil_reference": 0.08,
        "ndvi_vegetation_reference": 0.75,
        "canopy_regression_a": 130.0,
        "canopy_regression_b": -18.0,
        "typical_agb_range": (80.0, 250.0),
        "typical_height_range": (12.0, 35.0),
        "degradation_threshold": 20.0,
        "phenological_profile": [
            0.90, 0.92, 0.90, 0.82, 0.70, 0.60,
            0.55, 0.52, 0.55, 0.65, 0.78, 0.85,
        ],
    },

    # ------------------------------------------------------------------
    # 3. Tropical Coniferous Forest
    # ------------------------------------------------------------------
    # Central American pine-oak forests (Honduras, Guatemala, Mexico).
    # Moderate seasonality, fire-adapted understorey.
    "tropical_coniferous": {
        "ndvi_forest_threshold": 0.42,
        "ndvi_soil_reference": 0.07,
        "ndvi_vegetation_reference": 0.72,
        "canopy_regression_a": 125.0,
        "canopy_regression_b": -15.0,
        "typical_agb_range": (60.0, 200.0),
        "typical_height_range": (15.0, 35.0),
        "degradation_threshold": 18.0,
        "phenological_profile": [
            0.82, 0.80, 0.78, 0.80, 0.85, 0.92,
            0.95, 0.96, 0.94, 0.90, 0.86, 0.84,
        ],
    },

    # ------------------------------------------------------------------
    # 4. Temperate Broadleaf and Mixed Forest
    # ------------------------------------------------------------------
    # Europe, Eastern North America, East Asia deciduous zones.
    # Strong seasonal cycle with leaf-off winter period.
    "temperate_broadleaf_mixed": {
        "ndvi_forest_threshold": 0.40,
        "ndvi_soil_reference": 0.06,
        "ndvi_vegetation_reference": 0.82,
        "canopy_regression_a": 135.0,
        "canopy_regression_b": -17.0,
        "typical_agb_range": (100.0, 300.0),
        "typical_height_range": (15.0, 40.0),
        "degradation_threshold": 20.0,
        "phenological_profile": [
            0.35, 0.35, 0.45, 0.65, 0.85, 0.95,
            0.98, 0.96, 0.85, 0.65, 0.45, 0.35,
        ],
    },

    # ------------------------------------------------------------------
    # 5. Temperate Coniferous Forest
    # ------------------------------------------------------------------
    # Pacific Northwest, Scandinavia, Siberian taiga margins.
    # Evergreen canopy with mild seasonal NDVI modulation.
    "temperate_coniferous": {
        "ndvi_forest_threshold": 0.38,
        "ndvi_soil_reference": 0.05,
        "ndvi_vegetation_reference": 0.78,
        "canopy_regression_a": 128.0,
        "canopy_regression_b": -14.0,
        "typical_agb_range": (80.0, 350.0),
        "typical_height_range": (15.0, 60.0),
        "degradation_threshold": 18.0,
        "phenological_profile": [
            0.65, 0.65, 0.68, 0.75, 0.85, 0.92,
            0.95, 0.94, 0.88, 0.78, 0.70, 0.65,
        ],
    },

    # ------------------------------------------------------------------
    # 6. Boreal / Taiga
    # ------------------------------------------------------------------
    # Canada, Siberia, Fennoscandia. Low NDVI baseline due to
    # coniferous dominance and snow cover seasonality.
    "boreal_taiga": {
        "ndvi_forest_threshold": 0.30,
        "ndvi_soil_reference": 0.04,
        "ndvi_vegetation_reference": 0.65,
        "canopy_regression_a": 120.0,
        "canopy_regression_b": -10.0,
        "typical_agb_range": (30.0, 150.0),
        "typical_height_range": (8.0, 25.0),
        "degradation_threshold": 15.0,
        "phenological_profile": [
            0.25, 0.25, 0.30, 0.45, 0.70, 0.88,
            0.95, 0.90, 0.72, 0.45, 0.30, 0.25,
        ],
    },

    # ------------------------------------------------------------------
    # 7. Tropical Grassland and Savanna
    # ------------------------------------------------------------------
    # Brazilian Cerrado, African savanna, Llanos. Grass-tree mosaic
    # with strong wet/dry seasonality.
    "tropical_grassland_savanna": {
        "ndvi_forest_threshold": 0.35,
        "ndvi_soil_reference": 0.10,
        "ndvi_vegetation_reference": 0.70,
        "canopy_regression_a": 115.0,
        "canopy_regression_b": -12.0,
        "typical_agb_range": (20.0, 80.0),
        "typical_height_range": (3.0, 15.0),
        "degradation_threshold": 25.0,
        "phenological_profile": [
            0.85, 0.90, 0.88, 0.75, 0.55, 0.40,
            0.35, 0.32, 0.38, 0.55, 0.72, 0.82,
        ],
    },

    # ------------------------------------------------------------------
    # 8. Temperate Grassland
    # ------------------------------------------------------------------
    # Argentine Pampas, North American prairies. Low woody cover,
    # seasonal C3/C4 grass dominance shifts.
    "temperate_grassland": {
        "ndvi_forest_threshold": 0.35,
        "ndvi_soil_reference": 0.08,
        "ndvi_vegetation_reference": 0.65,
        "canopy_regression_a": 110.0,
        "canopy_regression_b": -10.0,
        "typical_agb_range": (5.0, 40.0),
        "typical_height_range": (1.0, 8.0),
        "degradation_threshold": 30.0,
        "phenological_profile": [
            0.40, 0.38, 0.42, 0.60, 0.80, 0.92,
            0.95, 0.90, 0.78, 0.55, 0.42, 0.38,
        ],
    },

    # ------------------------------------------------------------------
    # 9. Montane Grassland
    # ------------------------------------------------------------------
    # East African highlands, Ethiopian coffee zone, Andean paramo.
    # Altitude-reduced NDVI, bimodal rainfall in equatorial regions.
    "montane_grassland": {
        "ndvi_forest_threshold": 0.38,
        "ndvi_soil_reference": 0.06,
        "ndvi_vegetation_reference": 0.68,
        "canopy_regression_a": 118.0,
        "canopy_regression_b": -12.0,
        "typical_agb_range": (15.0, 60.0),
        "typical_height_range": (2.0, 12.0),
        "degradation_threshold": 22.0,
        "phenological_profile": [
            0.70, 0.68, 0.75, 0.85, 0.90, 0.82,
            0.75, 0.72, 0.78, 0.85, 0.82, 0.72,
        ],
    },

    # ------------------------------------------------------------------
    # 10. Mediterranean Woodland
    # ------------------------------------------------------------------
    # Mediterranean basin, California chaparral, Chilean matorral.
    # Summer drought stress, sclerophyllous vegetation.
    "mediterranean_woodland": {
        "ndvi_forest_threshold": 0.35,
        "ndvi_soil_reference": 0.08,
        "ndvi_vegetation_reference": 0.68,
        "canopy_regression_a": 118.0,
        "canopy_regression_b": -12.0,
        "typical_agb_range": (30.0, 120.0),
        "typical_height_range": (5.0, 20.0),
        "degradation_threshold": 22.0,
        "phenological_profile": [
            0.78, 0.82, 0.88, 0.90, 0.82, 0.65,
            0.50, 0.48, 0.55, 0.65, 0.72, 0.75,
        ],
    },

    # ------------------------------------------------------------------
    # 11. Mangrove
    # ------------------------------------------------------------------
    # Coastal tropical/subtropical tidal forests. Distinctive spectral
    # signature due to waterlogged substrate. High carbon density.
    "mangrove": {
        "ndvi_forest_threshold": 0.35,
        "ndvi_soil_reference": 0.00,
        "ndvi_vegetation_reference": 0.72,
        "canopy_regression_a": 122.0,
        "canopy_regression_b": -10.0,
        "typical_agb_range": (80.0, 300.0),
        "typical_height_range": (5.0, 30.0),
        "degradation_threshold": 15.0,
        "phenological_profile": [
            0.88, 0.88, 0.90, 0.92, 0.93, 0.92,
            0.90, 0.88, 0.87, 0.88, 0.88, 0.88,
        ],
    },

    # ------------------------------------------------------------------
    # 12. Flooded Grassland
    # ------------------------------------------------------------------
    # Pantanal, Sudd, Okavango Delta. Seasonal inundation creates
    # mixed water/vegetation spectral responses.
    "flooded_grassland": {
        "ndvi_forest_threshold": 0.30,
        "ndvi_soil_reference": 0.02,
        "ndvi_vegetation_reference": 0.65,
        "canopy_regression_a": 110.0,
        "canopy_regression_b": -8.0,
        "typical_agb_range": (10.0, 50.0),
        "typical_height_range": (1.0, 8.0),
        "degradation_threshold": 25.0,
        "phenological_profile": [
            0.80, 0.85, 0.88, 0.82, 0.65, 0.50,
            0.42, 0.40, 0.48, 0.60, 0.72, 0.78,
        ],
    },

    # ------------------------------------------------------------------
    # 13. Desert and Xeric Shrubland
    # ------------------------------------------------------------------
    # Sahel margins, Caatinga, arid woodland transitions.
    # Very low baseline NDVI, ephemeral green-up after rain events.
    "desert_xeric": {
        "ndvi_forest_threshold": 0.25,
        "ndvi_soil_reference": 0.05,
        "ndvi_vegetation_reference": 0.45,
        "canopy_regression_a": 100.0,
        "canopy_regression_b": -5.0,
        "typical_agb_range": (2.0, 25.0),
        "typical_height_range": (1.0, 6.0),
        "degradation_threshold": 30.0,
        "phenological_profile": [
            0.30, 0.28, 0.30, 0.35, 0.40, 0.38,
            0.55, 0.72, 0.65, 0.45, 0.35, 0.30,
        ],
    },

    # ------------------------------------------------------------------
    # 14. Tropical Plantation
    # ------------------------------------------------------------------
    # Oil palm and rubber monoculture plantations. High NDVI from
    # dense canopy but distinct spectral signature from natural forest.
    # Important for EUDR: plantations are NOT forest per EUDR Article 2.
    "tropical_plantation": {
        "ndvi_forest_threshold": 0.45,
        "ndvi_soil_reference": 0.06,
        "ndvi_vegetation_reference": 0.80,
        "canopy_regression_a": 132.0,
        "canopy_regression_b": -16.0,
        "typical_agb_range": (40.0, 150.0),
        "typical_height_range": (8.0, 25.0),
        "degradation_threshold": 20.0,
        "phenological_profile": [
            0.90, 0.91, 0.92, 0.93, 0.92, 0.90,
            0.88, 0.87, 0.88, 0.89, 0.90, 0.90,
        ],
    },

    # ------------------------------------------------------------------
    # 15. Agroforestry System
    # ------------------------------------------------------------------
    # Shade-grown coffee, cocoa under native canopy, timber agroforestry.
    # Mixed spectral response blending crop and native tree signatures.
    # EUDR: Agroforestry with >10% native canopy IS forest per EUDR.
    "agroforestry_system": {
        "ndvi_forest_threshold": 0.40,
        "ndvi_soil_reference": 0.08,
        "ndvi_vegetation_reference": 0.75,
        "canopy_regression_a": 125.0,
        "canopy_regression_b": -14.0,
        "typical_agb_range": (30.0, 120.0),
        "typical_height_range": (5.0, 20.0),
        "degradation_threshold": 20.0,
        "phenological_profile": [
            0.85, 0.87, 0.90, 0.92, 0.90, 0.85,
            0.80, 0.78, 0.80, 0.82, 0.84, 0.85,
        ],
    },

    # ------------------------------------------------------------------
    # 16. Degraded / Secondary Forest
    # ------------------------------------------------------------------
    # Regenerating forest after disturbance, logged areas, secondary
    # regrowth. Lower AGB and canopy height than primary forest.
    "degraded_secondary": {
        "ndvi_forest_threshold": 0.35,
        "ndvi_soil_reference": 0.08,
        "ndvi_vegetation_reference": 0.70,
        "canopy_regression_a": 118.0,
        "canopy_regression_b": -12.0,
        "typical_agb_range": (25.0, 150.0),
        "typical_height_range": (5.0, 25.0),
        "degradation_threshold": 18.0,
        "phenological_profile": [
            0.82, 0.84, 0.86, 0.88, 0.86, 0.82,
            0.78, 0.76, 0.78, 0.80, 0.81, 0.82,
        ],
    },
}


# ---------------------------------------------------------------------------
# Biome list
# ---------------------------------------------------------------------------

BIOME_LIST: List[str] = sorted(BIOME_PARAMETERS.keys())
"""Sorted list of all 16 recognized biome names."""


# ---------------------------------------------------------------------------
# Commodity-to-biome mapping
# ---------------------------------------------------------------------------
#
# Maps each EUDR-regulated commodity to its typical biome types.
# Used for automatic biome selection when the biome is not explicitly
# provided by the operator. A commodity may span multiple biomes
# depending on the producing region.

COMMODITY_BIOME_MAP: Dict[str, List[str]] = {
    "CATTLE": [
        "tropical_grassland_savanna",
        "tropical_dry_broadleaf",
        "temperate_grassland",
    ],
    "COCOA": [
        "tropical_moist_broadleaf",
        "agroforestry_system",
    ],
    "COFFEE": [
        "tropical_moist_broadleaf",
        "agroforestry_system",
        "montane_grassland",
    ],
    "OIL_PALM": [
        "tropical_moist_broadleaf",
        "tropical_plantation",
    ],
    "RUBBER": [
        "tropical_moist_broadleaf",
        "tropical_plantation",
    ],
    "SOYA": [
        "tropical_grassland_savanna",
        "tropical_dry_broadleaf",
        "temperate_grassland",
    ],
    "WOOD": [
        "tropical_moist_broadleaf",
        "tropical_dry_broadleaf",
        "tropical_coniferous",
        "temperate_broadleaf_mixed",
        "temperate_coniferous",
        "boreal_taiga",
        "mangrove",
    ],
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_biome_params(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve the full parameter set for a specific biome.

    Args:
        biome: Biome name (e.g., 'tropical_moist_broadleaf',
            'boreal_taiga'). Case-sensitive, must match a key
            in BIOME_PARAMETERS.

    Returns:
        Dictionary of biome parameters, or None if the biome
        name is not recognized.

    Example:
        >>> params = get_biome_params("tropical_moist_broadleaf")
        >>> params["ndvi_forest_threshold"]
        0.50
    """
    return BIOME_PARAMETERS.get(biome)


def get_biome_ndvi_threshold(biome: str) -> Optional[float]:
    """Get the forest NDVI threshold for a specific biome.

    Convenience function that extracts only the ndvi_forest_threshold
    from the biome parameter set.

    Args:
        biome: Biome name.

    Returns:
        NDVI threshold value, or None if biome is not recognized.
    """
    params = BIOME_PARAMETERS.get(biome)
    if params is None:
        return None
    return params["ndvi_forest_threshold"]


def get_dimidiation_refs(
    biome: str,
) -> Optional[Tuple[float, float]]:
    """Get the soil and vegetation NDVI references for dimidiation model.

    The dimidiation model computes fractional vegetation cover as:
        FVC = ((NDVI - NDVIs) / (NDVIv - NDVIs))^2

    Args:
        biome: Biome name.

    Returns:
        Tuple of (ndvi_soil_reference, ndvi_vegetation_reference),
        or None if biome is not recognized.
    """
    params = BIOME_PARAMETERS.get(biome)
    if params is None:
        return None
    return (
        params["ndvi_soil_reference"],
        params["ndvi_vegetation_reference"],
    )


def get_phenological_profile(
    biome: str,
) -> Optional[List[float]]:
    """Get the 12-month phenological NDVI profile for a biome.

    Args:
        biome: Biome name.

    Returns:
        List of 12 monthly NDVI fraction values (Jan=index 0),
        or None if biome is not recognized.
    """
    params = BIOME_PARAMETERS.get(biome)
    if params is None:
        return None
    return list(params["phenological_profile"])


def get_biomes_for_commodity(commodity: str) -> List[str]:
    """Get the list of typical biomes for an EUDR commodity.

    Args:
        commodity: EUDR commodity identifier (case-insensitive).
            One of: CATTLE, COCOA, COFFEE, OIL_PALM, RUBBER,
            SOYA, WOOD.

    Returns:
        List of biome name strings, or empty list if the
        commodity is not recognized.
    """
    return list(COMMODITY_BIOME_MAP.get(commodity.upper(), []))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BIOME_PARAMETERS",
    "BIOME_LIST",
    "COMMODITY_BIOME_MAP",
    "get_biome_params",
    "get_biome_ndvi_threshold",
    "get_dimidiation_refs",
    "get_phenological_profile",
    "get_biomes_for_commodity",
]
