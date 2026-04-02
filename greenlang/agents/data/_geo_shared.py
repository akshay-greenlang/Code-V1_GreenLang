# -*- coding: utf-8 -*-
"""
Shared Geospatial Enums and Models
===================================

Canonical definitions for geospatial enumerations and models shared across
the satellite remote sensing agent, deforestation satellite connector, and
GIS/mapping connector agents.

Each enum is the UNION of all values previously defined independently in:
    - satellite_remote_sensing_agent.py  (GL-DATA-X-007)
    - deforestation_satellite/models.py  (GL-DATA-GEO-003)
    - gis_connector/models.py           (GL-DATA-006)

Agents MUST import these enums from this module to avoid drift.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from greenlang.schemas import GreenLangBase


# =============================================================================
# Satellite Provider / Source
# =============================================================================

class SatelliteProvider(str, Enum):
    """Unified satellite data providers / sources.

    Union of SatelliteProvider (satellite_remote_sensing_agent) and
    SatelliteSource (deforestation_satellite).
    """

    # From satellite_remote_sensing_agent
    SENTINEL_2 = "sentinel_2"
    SENTINEL_1 = "sentinel_1"
    LANDSAT_8 = "landsat_8"
    LANDSAT_9 = "landsat_9"
    PLANET = "planet"
    MAXAR = "maxar"
    COPERNICUS = "copernicus"
    NASA_MODIS = "nasa_modis"
    SIMULATED = "simulated"
    # From deforestation_satellite (additional values)
    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    MODIS = "modis"
    HARMONIZED = "harmonized"


# Alias for backward compatibility with deforestation_satellite code
SatelliteSource = SatelliteProvider


# =============================================================================
# Vegetation Index
# =============================================================================

class VegetationIndex(str, Enum):
    """Spectral vegetation indices computable from satellite imagery.

    Union of indices from satellite_remote_sensing_agent and
    deforestation_satellite to avoid duplicate definitions.
    """

    NDVI = "ndvi"    # Normalized Difference Vegetation Index
    EVI = "evi"      # Enhanced Vegetation Index
    SAVI = "savi"    # Soil Adjusted Vegetation Index
    LAI = "lai"      # Leaf Area Index
    NDWI = "ndwi"    # Normalized Difference Water Index
    NBR = "nbr"      # Normalized Burn Ratio
    NDMI = "ndmi"    # Normalized Difference Moisture Index
    MSAVI = "msavi"  # Modified Soil Adjusted Vegetation Index


# =============================================================================
# Land Cover
# =============================================================================

class LandCoverClass(str, Enum):
    """Land cover classification (satellite / deforestation agents).

    Union of LandCoverClass from satellite_remote_sensing_agent and
    deforestation_satellite.
    """

    # From satellite_remote_sensing_agent
    FOREST = "forest"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    WETLAND = "wetland"
    URBAN = "urban"
    BARREN = "barren"
    WATER = "water"
    SHRUBLAND = "shrubland"
    SNOW_ICE = "snow_ice"
    # From deforestation_satellite (additional values)
    DENSE_FOREST = "dense_forest"
    OPEN_FOREST = "open_forest"
    BARE_SOIL = "bare_soil"
    UNKNOWN = "unknown"


class LandCoverType(str, Enum):
    """CORINE-derived land cover types (GIS connector).

    Union of LandCoverType from gis_connector and overlapping values
    from satellite agents.
    """

    URBAN = "urban"
    FOREST = "forest"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    WATER = "water"
    BARREN = "barren"
    SNOW_ICE = "snow_ice"
    SHRUBLAND = "shrubland"
    MANGROVE = "mangrove"


# =============================================================================
# Change Type
# =============================================================================

class ChangeType(str, Enum):
    """Land use/cover change types detected from satellite imagery.

    Union of change types from satellite_remote_sensing_agent (NBS monitoring)
    and deforestation_satellite (EUDR compliance).
    """

    # Common
    NO_CHANGE = "no_change"
    DEGRADATION = "degradation"
    # From satellite_remote_sensing_agent (NBS)
    DEFORESTATION = "deforestation"
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    URBANIZATION = "urbanization"
    AGRICULTURAL_EXPANSION = "agricultural_expansion"
    # From deforestation_satellite (EUDR)
    CLEAR_CUT = "clear_cut"
    PARTIAL_LOSS = "partial_loss"
    REGROWTH = "regrowth"


# =============================================================================
# Bounding Box (shared coordinate model)
# =============================================================================

class GeoBoundingBox(GreenLangBase):
    """Geographic bounding box with WGS84 coordinate constraints."""

    min_lat: float = Field(..., ge=-90, le=90)
    max_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lon: float = Field(..., ge=-180, le=180)
