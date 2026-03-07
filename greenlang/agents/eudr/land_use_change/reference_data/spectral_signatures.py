# -*- coding: utf-8 -*-
"""
Spectral Signatures for Land Use Types - AGENT-EUDR-005

Provides reference spectral signatures for each land use category used by
the LandUseClassifier engine. Signatures include Sentinel-2 band values,
vegetation index ranges, texture features (GLCM), and phenology patterns.

Biome-adjusted signatures account for spectral variations across 16 global
biome types. Commodity-specific profiles distinguish young vs mature
plantations for EUDR-regulated crops.

Data Sources:
    - Sentinel-2 MSI Level-2A radiometric calibration data
    - Copernicus Global Land Cover Layers (CGLS-LC100)
    - Potapov et al. (2022) - Tropical plantation mapping
    - Huete et al. (2002) - EVI biome characterization
    - Haralick (1973) - GLCM texture feature definitions
    - ESA WorldCover 10m (2021) validation datasets

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent
Status: Production Ready
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.land_use_change.reference_data.land_use_parameters import (
    LandUseCategory,
)


# ---------------------------------------------------------------------------
# Reference spectral signatures per land use category
# ---------------------------------------------------------------------------
#
# Band values are Sentinel-2 MSI Level-2A bottom-of-atmosphere reflectance
# (dimensionless, 0.0 to 1.0 scale). Ranges are (typical_min, typical_max).
#
# Bands:
#   B2  = Blue (490nm, 10m)
#   B3  = Green (560nm, 10m)
#   B4  = Red (665nm, 10m)
#   B5  = Red Edge 1 (705nm, 20m)
#   B6  = Red Edge 2 (740nm, 20m)
#   B7  = Red Edge 3 (783nm, 20m)
#   B8  = NIR (842nm, 10m)
#   B8A = NIR narrow (865nm, 20m)
#   B11 = SWIR 1 (1610nm, 20m)
#   B12 = SWIR 2 (2190nm, 20m)

SPECTRAL_SIGNATURES: Dict[LandUseCategory, Dict[str, Any]] = {
    LandUseCategory.FOREST_LAND: {
        "band_values": {
            "B2": (0.01, 0.04),
            "B3": (0.02, 0.06),
            "B4": (0.01, 0.04),
            "B5": (0.04, 0.10),
            "B6": (0.15, 0.35),
            "B7": (0.20, 0.40),
            "B8": (0.20, 0.45),
            "B8A": (0.22, 0.45),
            "B11": (0.06, 0.18),
            "B12": (0.03, 0.10),
        },
        "ndvi_range": (0.45, 0.92),
        "evi_range": (0.28, 0.58),
        "ndmi_range": (0.15, 0.55),
        "savi_range": (0.30, 0.65),
        "texture_features": {
            "homogeneity": (0.25, 0.55),
            "contrast": (50.0, 350.0),
            "correlation": (0.70, 0.95),
            "dissimilarity": (3.0, 12.0),
        },
        "phenology_pattern": (
            "Evergreen tropical: minimal seasonal NDVI variation (amplitude <0.10). "
            "Deciduous temperate: annual cycle with leaf-on peak in June-August "
            "(Northern Hemisphere) and leaf-off trough in December-February."
        ),
    },
    LandUseCategory.CROPLAND: {
        "band_values": {
            "B2": (0.02, 0.08),
            "B3": (0.03, 0.10),
            "B4": (0.02, 0.10),
            "B5": (0.04, 0.12),
            "B6": (0.10, 0.30),
            "B7": (0.12, 0.35),
            "B8": (0.15, 0.45),
            "B8A": (0.15, 0.45),
            "B11": (0.10, 0.30),
            "B12": (0.05, 0.20),
        },
        "ndvi_range": (0.20, 0.85),
        "evi_range": (0.12, 0.55),
        "ndmi_range": (-0.05, 0.35),
        "savi_range": (0.15, 0.60),
        "texture_features": {
            "homogeneity": (0.35, 0.70),
            "contrast": (20.0, 200.0),
            "correlation": (0.60, 0.90),
            "dissimilarity": (2.0, 8.0),
        },
        "phenology_pattern": (
            "Strong seasonal cycle with rapid green-up at planting, peak NDVI "
            "at canopy closure, and rapid senescence at harvest. Annual crops "
            "show 1-2 distinct cycles. Fallow periods exhibit bare soil signature."
        ),
    },
    LandUseCategory.GRASSLAND: {
        "band_values": {
            "B2": (0.03, 0.08),
            "B3": (0.04, 0.10),
            "B4": (0.03, 0.10),
            "B5": (0.05, 0.12),
            "B6": (0.10, 0.25),
            "B7": (0.12, 0.30),
            "B8": (0.15, 0.40),
            "B8A": (0.15, 0.38),
            "B11": (0.10, 0.28),
            "B12": (0.06, 0.18),
        },
        "ndvi_range": (0.15, 0.65),
        "evi_range": (0.08, 0.40),
        "ndmi_range": (-0.10, 0.25),
        "savi_range": (0.10, 0.45),
        "texture_features": {
            "homogeneity": (0.45, 0.80),
            "contrast": (10.0, 120.0),
            "correlation": (0.50, 0.85),
            "dissimilarity": (1.5, 6.0),
        },
        "phenology_pattern": (
            "Moderate seasonal cycle tied to rainfall in tropical regions and "
            "temperature in temperate regions. Peak greenness during wet season "
            "or summer, brown-down during dry season or winter. Managed pastures "
            "show more uniform greenness due to irrigation."
        ),
    },
    LandUseCategory.WETLAND: {
        "band_values": {
            "B2": (0.02, 0.06),
            "B3": (0.02, 0.07),
            "B4": (0.01, 0.05),
            "B5": (0.03, 0.10),
            "B6": (0.08, 0.25),
            "B7": (0.10, 0.30),
            "B8": (0.08, 0.35),
            "B8A": (0.08, 0.32),
            "B11": (0.02, 0.12),
            "B12": (0.01, 0.08),
        },
        "ndvi_range": (0.10, 0.65),
        "evi_range": (0.05, 0.42),
        "ndmi_range": (0.25, 0.75),
        "savi_range": (0.08, 0.45),
        "texture_features": {
            "homogeneity": (0.30, 0.65),
            "contrast": (30.0, 250.0),
            "correlation": (0.55, 0.88),
            "dissimilarity": (2.5, 10.0),
        },
        "phenology_pattern": (
            "Seasonal variation driven by water level fluctuations. Mangroves "
            "show stable high NDVI year-round. Peatlands exhibit moderate "
            "seasonal variation. Seasonally flooded wetlands show sharp NDVI "
            "drops during inundation periods."
        ),
    },
    LandUseCategory.SETTLEMENT: {
        "band_values": {
            "B2": (0.05, 0.15),
            "B3": (0.06, 0.16),
            "B4": (0.06, 0.18),
            "B5": (0.07, 0.18),
            "B6": (0.08, 0.20),
            "B7": (0.08, 0.22),
            "B8": (0.10, 0.30),
            "B8A": (0.10, 0.28),
            "B11": (0.12, 0.35),
            "B12": (0.08, 0.28),
        },
        "ndvi_range": (-0.05, 0.25),
        "evi_range": (-0.03, 0.18),
        "ndmi_range": (-0.35, 0.00),
        "savi_range": (-0.05, 0.18),
        "texture_features": {
            "homogeneity": (0.15, 0.40),
            "contrast": (200.0, 800.0),
            "correlation": (0.30, 0.65),
            "dissimilarity": (8.0, 20.0),
        },
        "phenology_pattern": (
            "Minimal seasonal variation. Built-up surfaces show stable high "
            "SWIR and low NDVI year-round. Urban vegetation (parks, gardens) "
            "adds slight seasonal signal but overall amplitude is <0.05."
        ),
    },
    LandUseCategory.OTHER_LAND: {
        "band_values": {
            "B2": (0.08, 0.25),
            "B3": (0.10, 0.28),
            "B4": (0.10, 0.30),
            "B5": (0.10, 0.28),
            "B6": (0.10, 0.28),
            "B7": (0.10, 0.28),
            "B8": (0.12, 0.32),
            "B8A": (0.12, 0.30),
            "B11": (0.18, 0.45),
            "B12": (0.12, 0.38),
        },
        "ndvi_range": (-0.15, 0.10),
        "evi_range": (-0.08, 0.08),
        "ndmi_range": (-0.45, -0.10),
        "savi_range": (-0.10, 0.08),
        "texture_features": {
            "homogeneity": (0.50, 0.85),
            "contrast": (5.0, 80.0),
            "correlation": (0.40, 0.75),
            "dissimilarity": (1.0, 5.0),
        },
        "phenology_pattern": (
            "No seasonal variation. Bare soil and rock surfaces show stable "
            "high SWIR reflectance and near-zero or negative NDVI throughout "
            "the year. Sand dunes may show minor spectral shifts with moisture."
        ),
    },
    LandUseCategory.OIL_PALM_PLANTATION: {
        "band_values": {
            "B2": (0.01, 0.04),
            "B3": (0.02, 0.06),
            "B4": (0.01, 0.05),
            "B5": (0.04, 0.10),
            "B6": (0.12, 0.30),
            "B7": (0.18, 0.38),
            "B8": (0.20, 0.45),
            "B8A": (0.20, 0.42),
            "B11": (0.08, 0.22),
            "B12": (0.04, 0.12),
        },
        "ndvi_range": (0.40, 0.82),
        "evi_range": (0.22, 0.50),
        "ndmi_range": (0.10, 0.40),
        "savi_range": (0.28, 0.58),
        "texture_features": {
            "homogeneity": (0.30, 0.55),
            "contrast": (80.0, 400.0),
            "correlation": (0.65, 0.90),
            "dissimilarity": (4.0, 14.0),
        },
        "phenology_pattern": (
            "Low seasonal amplitude due to evergreen canopy. Regular planting "
            "geometry creates distinctive texture patterns. Young plantations "
            "(0-3yr) show lower NDVI (0.30-0.55) with exposed soil between "
            "palms. Mature plantations (8+ yr) reach NDVI 0.70-0.82."
        ),
    },
    LandUseCategory.RUBBER_PLANTATION: {
        "band_values": {
            "B2": (0.02, 0.05),
            "B3": (0.03, 0.07),
            "B4": (0.02, 0.06),
            "B5": (0.04, 0.11),
            "B6": (0.10, 0.28),
            "B7": (0.15, 0.35),
            "B8": (0.18, 0.42),
            "B8A": (0.18, 0.40),
            "B11": (0.08, 0.24),
            "B12": (0.04, 0.14),
        },
        "ndvi_range": (0.30, 0.80),
        "evi_range": (0.18, 0.48),
        "ndmi_range": (0.05, 0.38),
        "savi_range": (0.22, 0.55),
        "texture_features": {
            "homogeneity": (0.28, 0.52),
            "contrast": (70.0, 380.0),
            "correlation": (0.62, 0.88),
            "dissimilarity": (3.5, 13.0),
        },
        "phenology_pattern": (
            "Distinctive wintering period with leaf-shedding in dry season "
            "causes NDVI drops of 0.15-0.30 (January-March in Southeast "
            "Asia). This defoliation-refoliation cycle is the strongest "
            "spectral discriminator from natural evergreen forest."
        ),
    },
    LandUseCategory.PLANTATION_FOREST: {
        "band_values": {
            "B2": (0.01, 0.04),
            "B3": (0.02, 0.06),
            "B4": (0.01, 0.05),
            "B5": (0.04, 0.10),
            "B6": (0.13, 0.32),
            "B7": (0.18, 0.38),
            "B8": (0.20, 0.45),
            "B8A": (0.20, 0.43),
            "B11": (0.07, 0.20),
            "B12": (0.03, 0.12),
        },
        "ndvi_range": (0.40, 0.85),
        "evi_range": (0.25, 0.55),
        "ndmi_range": (0.10, 0.45),
        "savi_range": (0.28, 0.60),
        "texture_features": {
            "homogeneity": (0.35, 0.60),
            "contrast": (40.0, 280.0),
            "correlation": (0.72, 0.95),
            "dissimilarity": (2.5, 10.0),
        },
        "phenology_pattern": (
            "More uniform canopy than natural forest due to single-species "
            "or dual-species composition. Eucalyptus plantations show "
            "moderate seasonal variation. Pine plantations are evergreen "
            "with low amplitude. Clear-cut harvesting creates abrupt NDVI "
            "drops followed by regrowth."
        ),
    },
    LandUseCategory.WATER_BODY: {
        "band_values": {
            "B2": (0.03, 0.12),
            "B3": (0.02, 0.10),
            "B4": (0.01, 0.06),
            "B5": (0.01, 0.04),
            "B6": (0.00, 0.03),
            "B7": (0.00, 0.02),
            "B8": (0.00, 0.05),
            "B8A": (0.00, 0.04),
            "B11": (0.00, 0.03),
            "B12": (0.00, 0.02),
        },
        "ndvi_range": (-0.50, -0.05),
        "evi_range": (-0.30, -0.02),
        "ndmi_range": (0.60, 1.00),
        "savi_range": (-0.35, -0.03),
        "texture_features": {
            "homogeneity": (0.70, 0.95),
            "contrast": (2.0, 30.0),
            "correlation": (0.80, 0.98),
            "dissimilarity": (0.5, 3.0),
        },
        "phenology_pattern": (
            "Minimal seasonal variation in deep water bodies. Shallow lakes "
            "and reservoirs may show seasonal turbidity changes and minor "
            "reflectance shifts. Algal blooms can temporarily increase "
            "green-band reflectance."
        ),
    },
}


# ---------------------------------------------------------------------------
# Biome-adjusted spectral signatures
# ---------------------------------------------------------------------------
#
# Adjustments applied to the base signatures when a specific biome context
# is known. Each biome provides multiplicative scale factors for NDVI, EVI,
# and NDMI ranges, and additive offsets for band reflectance values.
#
# Format:
#   biome_name -> {
#       "ndvi_scale": float,  -- multiply NDVI range bounds by this factor
#       "evi_scale": float,   -- multiply EVI range bounds
#       "ndmi_offset": float, -- add to NDMI range bounds
#       "swir_offset": float, -- add to SWIR band ranges
#       "description": str,
#   }

BIOME_ADJUSTED_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "tropical_moist_broadleaf": {
        "ndvi_scale": 1.05,
        "evi_scale": 1.08,
        "ndmi_offset": 0.05,
        "swir_offset": -0.02,
        "description": (
            "Dense canopy with high moisture content. Highest EVI globally. "
            "Amazon, Congo Basin, Southeast Asian lowland forests."
        ),
    },
    "tropical_dry_broadleaf": {
        "ndvi_scale": 0.90,
        "evi_scale": 0.88,
        "ndmi_offset": -0.05,
        "swir_offset": 0.03,
        "description": (
            "Seasonal deciduous component. Strong dry-season NDVI depression. "
            "Chaco, Cerrado dry forests, Thai-Myanmar dry forests."
        ),
    },
    "tropical_coniferous": {
        "ndvi_scale": 0.92,
        "evi_scale": 0.90,
        "ndmi_offset": -0.02,
        "swir_offset": 0.01,
        "description": (
            "Tropical pine-dominated forests at higher elevations. "
            "Central American highlands, Luzon pine forests."
        ),
    },
    "temperate_broadleaf_mixed": {
        "ndvi_scale": 0.95,
        "evi_scale": 0.93,
        "ndmi_offset": -0.03,
        "swir_offset": 0.02,
        "description": (
            "Deciduous and mixed forests with strong annual cycle. "
            "Eastern North America, Western Europe, East Asia."
        ),
    },
    "temperate_coniferous": {
        "ndvi_scale": 0.88,
        "evi_scale": 0.85,
        "ndmi_offset": -0.02,
        "swir_offset": 0.02,
        "description": (
            "Evergreen coniferous forests. Pacific Northwest, Scandinavian "
            "taiga-boreal transition, Siberian larch-spruce forests."
        ),
    },
    "boreal_taiga": {
        "ndvi_scale": 0.80,
        "evi_scale": 0.75,
        "ndmi_offset": -0.05,
        "swir_offset": 0.03,
        "description": (
            "Low-productivity coniferous forests. Short growing season. "
            "Canadian Shield, Scandinavian taiga, Siberian boreal."
        ),
    },
    "tropical_subtropical_grassland": {
        "ndvi_scale": 0.85,
        "evi_scale": 0.82,
        "ndmi_offset": -0.08,
        "swir_offset": 0.05,
        "description": (
            "Grass-dominated with scattered trees. Cerrado, East African "
            "savannas, Australian tropical grasslands."
        ),
    },
    "temperate_grassland": {
        "ndvi_scale": 0.82,
        "evi_scale": 0.78,
        "ndmi_offset": -0.10,
        "swir_offset": 0.06,
        "description": (
            "Steppe and prairie grasslands. Argentine Pampas, Central Asian "
            "steppe, North American Great Plains."
        ),
    },
    "montane_grassland": {
        "ndvi_scale": 0.78,
        "evi_scale": 0.75,
        "ndmi_offset": -0.05,
        "swir_offset": 0.04,
        "description": (
            "High-altitude grasslands and paramo. Andean paramo, East "
            "African afroalpine, Tibetan Plateau meadows."
        ),
    },
    "flooded_grassland": {
        "ndvi_scale": 0.88,
        "evi_scale": 0.85,
        "ndmi_offset": 0.15,
        "swir_offset": -0.04,
        "description": (
            "Seasonally inundated grasslands. Pantanal, Everglades, "
            "Sudd wetlands, Okavango Delta."
        ),
    },
    "mangrove": {
        "ndvi_scale": 0.92,
        "evi_scale": 0.90,
        "ndmi_offset": 0.20,
        "swir_offset": -0.05,
        "description": (
            "Coastal salt-tolerant forests. High NDMI due to tidal "
            "influence. Sundarbans, Niger Delta, Mesoamerican Reef."
        ),
    },
    "desert_xeric_shrubland": {
        "ndvi_scale": 0.50,
        "evi_scale": 0.45,
        "ndmi_offset": -0.20,
        "swir_offset": 0.10,
        "description": (
            "Extremely arid with sparse vegetation. Sahara margins, "
            "Namib, Arabian Peninsula, Thar Desert."
        ),
    },
    "mediterranean_forest": {
        "ndvi_scale": 0.88,
        "evi_scale": 0.85,
        "ndmi_offset": -0.05,
        "swir_offset": 0.03,
        "description": (
            "Sclerophyllous forests and shrublands. Summer-dry, winter-wet "
            "cycle. Mediterranean Basin, California, Chile, SW Australia."
        ),
    },
    "peat_swamp_forest": {
        "ndvi_scale": 0.95,
        "evi_scale": 0.92,
        "ndmi_offset": 0.15,
        "swir_offset": -0.04,
        "description": (
            "Waterlogged forest on deep peat soils. High carbon density. "
            "Central Kalimantan, Sumatra, Sarawak peatlands."
        ),
    },
    "montane_cloud_forest": {
        "ndvi_scale": 0.93,
        "evi_scale": 0.90,
        "ndmi_offset": 0.10,
        "swir_offset": -0.02,
        "description": (
            "High-altitude forests with persistent cloud immersion. "
            "Andean yungas, East African highlands, SE Asian cloud forests."
        ),
    },
    "cerrado_savanna": {
        "ndvi_scale": 0.82,
        "evi_scale": 0.80,
        "ndmi_offset": -0.08,
        "swir_offset": 0.05,
        "description": (
            "Brazilian Cerrado woodland-savanna mosaic. Fire-adapted. "
            "Strong wet-dry seasonality with NDVI amplitude of 0.25-0.40."
        ),
    },
}


# ---------------------------------------------------------------------------
# Commodity spectral profiles
# ---------------------------------------------------------------------------
#
# Spectral profiles for EUDR-regulated commodity crops, with separate
# profiles for young (establishment) and mature (production) stages.

COMMODITY_SPECTRAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "palm_oil": {
        "young": {
            "age_range_years": (0, 3),
            "ndvi_range": (0.25, 0.55),
            "evi_range": (0.12, 0.35),
            "canopy_cover_pct": (10.0, 40.0),
            "row_spacing_m": (8.0, 9.0),
            "description": (
                "Young oil palm with exposed inter-row soil. Regular "
                "triangular planting pattern visible at 10m resolution. "
                "NDVI increases steadily with age."
            ),
        },
        "mature": {
            "age_range_years": (8, 25),
            "ndvi_range": (0.60, 0.82),
            "evi_range": (0.35, 0.50),
            "canopy_cover_pct": (70.0, 95.0),
            "row_spacing_m": (8.0, 9.0),
            "description": (
                "Mature oil palm with closed canopy. NDVI comparable to "
                "secondary forest but distinguishable by lower NDVI "
                "variability and regular texture pattern."
            ),
        },
    },
    "rubber": {
        "young": {
            "age_range_years": (0, 5),
            "ndvi_range": (0.20, 0.50),
            "evi_range": (0.10, 0.30),
            "canopy_cover_pct": (5.0, 35.0),
            "row_spacing_m": (3.0, 7.0),
            "description": (
                "Young rubber with significant bare ground between rows. "
                "Often intercropped with annual crops during establishment."
            ),
        },
        "mature": {
            "age_range_years": (7, 30),
            "ndvi_range": (0.55, 0.80),
            "evi_range": (0.30, 0.48),
            "canopy_cover_pct": (60.0, 90.0),
            "row_spacing_m": (3.0, 7.0),
            "description": (
                "Mature rubber with closed canopy. Distinctive wintering "
                "defoliation creates annual NDVI dip of 0.15-0.30 that "
                "distinguishes it from evergreen forest."
            ),
        },
    },
    "cocoa": {
        "young": {
            "age_range_years": (0, 3),
            "ndvi_range": (0.20, 0.50),
            "evi_range": (0.10, 0.30),
            "canopy_cover_pct": (10.0, 40.0),
            "row_spacing_m": (3.0, 4.0),
            "description": (
                "Young cocoa often planted under shade trees. Mixed "
                "spectral signal from cocoa seedlings and shade canopy."
            ),
        },
        "mature": {
            "age_range_years": (5, 30),
            "ndvi_range": (0.50, 0.78),
            "evi_range": (0.28, 0.48),
            "canopy_cover_pct": (55.0, 85.0),
            "row_spacing_m": (3.0, 4.0),
            "description": (
                "Mature cocoa agroforestry. Shade-grown cocoa has higher "
                "NDVI than full-sun cocoa. Difficult to distinguish from "
                "degraded secondary forest at 10m resolution."
            ),
        },
    },
    "coffee": {
        "young": {
            "age_range_years": (0, 3),
            "ndvi_range": (0.18, 0.45),
            "evi_range": (0.08, 0.28),
            "canopy_cover_pct": (8.0, 35.0),
            "row_spacing_m": (1.5, 3.0),
            "description": (
                "Young coffee bushes with significant bare soil. Higher "
                "planting density than tree crops. Often on sloped terrain."
            ),
        },
        "mature": {
            "age_range_years": (4, 20),
            "ndvi_range": (0.45, 0.72),
            "evi_range": (0.25, 0.45),
            "canopy_cover_pct": (45.0, 80.0),
            "row_spacing_m": (1.5, 3.0),
            "description": (
                "Mature coffee. Shade-grown shows higher NDVI than "
                "sun-grown. Harvest pruning creates periodic NDVI drops. "
                "Altitude-dependent spectral characteristics."
            ),
        },
    },
    "soya": {
        "young": {
            "age_range_years": (0, 0),
            "ndvi_range": (0.10, 0.30),
            "evi_range": (0.05, 0.18),
            "canopy_cover_pct": (0.0, 20.0),
            "row_spacing_m": (0.4, 0.8),
            "description": (
                "Early-season soya with bare soil between rows. Rapid "
                "green-up phase. Typically planted September-November "
                "in Brazil, May-June in North America."
            ),
        },
        "mature": {
            "age_range_years": (0, 0),
            "ndvi_range": (0.60, 0.88),
            "evi_range": (0.35, 0.55),
            "canopy_cover_pct": (80.0, 100.0),
            "row_spacing_m": (0.4, 0.8),
            "description": (
                "Peak-season soya with full canopy closure. Very high "
                "NDVI comparable to forest. Distinguished by sharp "
                "senescence and harvest within a single growing season."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_spectral_signature(
    land_use: LandUseCategory,
    biome: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve the spectral signature for a land use class, optionally biome-adjusted.

    When a biome is specified, the base spectral signature is adjusted using
    the biome-specific scale factors and offsets from BIOME_ADJUSTED_SIGNATURES.

    Args:
        land_use: The LandUseCategory to look up.
        biome: Optional biome name for adjusted signatures.

    Returns:
        Dictionary containing band values, vegetation index ranges,
        texture features, and phenology pattern.

    Raises:
        KeyError: If the land use category is not found.
    """
    if land_use not in SPECTRAL_SIGNATURES:
        raise KeyError(
            f"No spectral signature for land use: {land_use}. "
            f"Valid categories: {[c.value for c in LandUseCategory]}"
        )

    base_sig = SPECTRAL_SIGNATURES[land_use]

    if biome is None or biome not in BIOME_ADJUSTED_SIGNATURES:
        return dict(base_sig)

    adjustment = BIOME_ADJUSTED_SIGNATURES[biome]
    adjusted = dict(base_sig)

    # Adjust NDVI range
    ndvi_scale = adjustment.get("ndvi_scale", 1.0)
    base_ndvi = base_sig.get("ndvi_range", (0.0, 1.0))
    adjusted["ndvi_range"] = (
        max(-1.0, base_ndvi[0] * ndvi_scale),
        min(1.0, base_ndvi[1] * ndvi_scale),
    )

    # Adjust EVI range
    evi_scale = adjustment.get("evi_scale", 1.0)
    base_evi = base_sig.get("evi_range", (0.0, 1.0))
    adjusted["evi_range"] = (
        max(-1.0, base_evi[0] * evi_scale),
        min(1.0, base_evi[1] * evi_scale),
    )

    # Adjust NDMI range with offset
    ndmi_offset = adjustment.get("ndmi_offset", 0.0)
    base_ndmi = base_sig.get("ndmi_range", (0.0, 1.0))
    adjusted["ndmi_range"] = (
        max(-1.0, base_ndmi[0] + ndmi_offset),
        min(1.0, base_ndmi[1] + ndmi_offset),
    )

    # Adjust SWIR bands with offset
    swir_offset = adjustment.get("swir_offset", 0.0)
    if "band_values" in base_sig:
        adjusted_bands = dict(base_sig["band_values"])
        for band_key in ("B11", "B12"):
            if band_key in adjusted_bands:
                original = adjusted_bands[band_key]
                adjusted_bands[band_key] = (
                    max(0.0, original[0] + swir_offset),
                    max(0.0, original[1] + swir_offset),
                )
        adjusted["band_values"] = adjusted_bands

    return adjusted


def compute_spectral_distance(
    observed: Dict[str, float],
    reference: Dict[str, Tuple[float, float]],
) -> float:
    """Compute the normalized spectral distance between observed values and a reference range.

    For each index (NDVI, EVI, NDMI, SAVI), calculates the distance from
    the observed value to the reference range. If the observed value falls
    within the range, distance is 0. Otherwise, it is the normalized
    distance to the nearest range boundary.

    Uses Euclidean distance across all available matching indices.

    Args:
        observed: Dictionary mapping index names to observed float values.
            Keys should be in the form 'ndvi', 'evi', 'ndmi', 'savi'.
        reference: Dictionary mapping index names with '_range' suffix to
            (min, max) tuples. Keys should match observed with '_range' appended.

    Returns:
        Normalized Euclidean distance (0.0 = perfect match within all ranges,
        higher values indicate greater mismatch). Range is [0.0, inf).
    """
    squared_sum = 0.0
    count = 0

    for index_name, value in observed.items():
        range_key = f"{index_name}_range"
        if range_key not in reference:
            continue

        ref_min, ref_max = reference[range_key]
        ref_span = ref_max - ref_min
        if ref_span <= 0:
            continue

        if value < ref_min:
            dist = (ref_min - value) / ref_span
        elif value > ref_max:
            dist = (value - ref_max) / ref_span
        else:
            dist = 0.0

        squared_sum += dist * dist
        count += 1

    if count == 0:
        return float("inf")

    return math.sqrt(squared_sum / count)


def classify_by_spectral_distance(
    observed: Dict[str, float],
    all_references: Optional[Dict[LandUseCategory, Dict[str, Any]]] = None,
) -> Tuple[LandUseCategory, float]:
    """Classify an observed spectral sample by minimum distance to reference signatures.

    Computes the spectral distance from the observed values to every reference
    signature and returns the category with the minimum distance.

    Args:
        observed: Dictionary mapping index names to observed float values.
            Keys: 'ndvi', 'evi', 'ndmi', 'savi'.
        all_references: Optional override for reference signatures. If None,
            uses the module-level SPECTRAL_SIGNATURES dictionary.

    Returns:
        Tuple of (best_matching_category, distance_score). Lower distance
        indicates higher confidence in the classification.

    Raises:
        ValueError: If no valid references are available for comparison.
    """
    refs = all_references if all_references is not None else SPECTRAL_SIGNATURES

    if not refs:
        raise ValueError("No reference signatures available for classification")

    best_category: Optional[LandUseCategory] = None
    best_distance = float("inf")

    for category, signature in refs.items():
        distance = compute_spectral_distance(observed, signature)
        if distance < best_distance:
            best_distance = distance
            best_category = category

    if best_category is None:
        raise ValueError("Failed to classify: no valid distances computed")

    return (best_category, best_distance)


def get_commodity_profile(
    commodity: str,
    stage: str = "mature",
) -> Optional[Dict[str, Any]]:
    """Retrieve the spectral profile for an EUDR commodity crop at a given growth stage.

    Args:
        commodity: EUDR commodity name (lowercase). One of 'palm_oil',
            'rubber', 'cocoa', 'coffee', 'soya'.
        stage: Growth stage, either 'young' or 'mature'. Defaults to 'mature'.

    Returns:
        Dictionary with spectral profile data, or None if the commodity
        or stage is not found.
    """
    profile = COMMODITY_SPECTRAL_PROFILES.get(commodity.lower())
    if profile is None:
        return None
    return profile.get(stage)


def get_biome_adjustment(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve the biome adjustment parameters.

    Args:
        biome: Biome name (e.g., 'tropical_moist_broadleaf', 'cerrado_savanna').

    Returns:
        Dictionary with scale factors and offsets, or None if biome not found.
    """
    return BIOME_ADJUSTED_SIGNATURES.get(biome)


def get_all_biome_names() -> List[str]:
    """Return sorted list of all recognized biome names.

    Returns:
        Sorted list of biome name strings.
    """
    return sorted(BIOME_ADJUSTED_SIGNATURES.keys())


def get_all_commodity_names() -> List[str]:
    """Return sorted list of all commodity names with spectral profiles.

    Returns:
        Sorted list of commodity name strings.
    """
    return sorted(COMMODITY_SPECTRAL_PROFILES.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SPECTRAL_SIGNATURES",
    "BIOME_ADJUSTED_SIGNATURES",
    "COMMODITY_SPECTRAL_PROFILES",
    "get_spectral_signature",
    "compute_spectral_distance",
    "classify_by_spectral_distance",
    "get_commodity_profile",
    "get_biome_adjustment",
    "get_all_biome_names",
    "get_all_commodity_names",
]
