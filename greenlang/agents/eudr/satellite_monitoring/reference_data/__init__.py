# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-003: Satellite Monitoring Agent

Provides built-in reference datasets for satellite-based forest monitoring:
    - forest_thresholds: NDVI/EVI classification thresholds per biome
    - seasonal_baselines: Phenological NDVI adjustment profiles per region
    - satellite_specs: Sensor band specifications for all supported satellites

These datasets enable deterministic, zero-hallucination spectral analysis
without external API dependencies. All data is version-tracked and
provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
"""

from greenlang.agents.eudr.satellite_monitoring.reference_data.forest_thresholds import (
    BIOME_NDVI_THRESHOLDS,
    COMMODITY_BIOME_MAP,
    classify_ndvi,
    get_biome_for_commodity,
    get_forest_threshold,
)
from greenlang.agents.eudr.satellite_monitoring.reference_data.seasonal_baselines import (
    SEASONAL_NDVI_PROFILES,
    adjust_ndvi_for_season,
    get_seasonal_adjustment,
)
from greenlang.agents.eudr.satellite_monitoring.reference_data.satellite_specs import (
    SENTINEL2_BANDS,
    LANDSAT8_BANDS,
    SENTINEL1_MODES,
    get_band_for_index,
    get_resolution,
)

__all__ = [
    "BIOME_NDVI_THRESHOLDS",
    "COMMODITY_BIOME_MAP",
    "classify_ndvi",
    "get_biome_for_commodity",
    "get_forest_threshold",
    "SEASONAL_NDVI_PROFILES",
    "adjust_ndvi_for_season",
    "get_seasonal_adjustment",
    "SENTINEL2_BANDS",
    "LANDSAT8_BANDS",
    "SENTINEL1_MODES",
    "get_band_for_index",
    "get_resolution",
]
