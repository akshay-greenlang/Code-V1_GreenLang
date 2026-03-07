# -*- coding: utf-8 -*-
"""
Ocean Mask Reference Data - AGENT-EUDR-002

Provides simplified land/ocean classification for GPS coordinate validation.
Uses major ocean bounding regions and known inland water body exclusion zones
to determine if a coordinate is on land or water.

This is a simplified approximation. For production accuracy, use PostGIS
with Natural Earth or GSHHG coastline data.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Major ocean regions as bounding boxes: (lat_min, lat_max, lon_min, lon_max)
# These are simplified approximations of deep ocean areas where no
# EUDR-regulated agriculture is possible.
MAJOR_OCEAN_REGIONS: List[Dict[str, object]] = [
    # Mid-Atlantic (between Americas and Africa/Europe)
    {"name": "Mid-Atlantic North", "bbox": (10.0, 55.0, -50.0, -15.0)},
    {"name": "Mid-Atlantic Equatorial", "bbox": (-10.0, 10.0, -40.0, -10.0)},
    {"name": "Mid-Atlantic South", "bbox": (-55.0, -10.0, -45.0, -5.0)},
    # Central Pacific
    {"name": "Central Pacific North", "bbox": (5.0, 45.0, -175.0, -120.0)},
    {"name": "Central Pacific South", "bbox": (-45.0, 5.0, -175.0, -90.0)},
    {"name": "Western Pacific", "bbox": (-10.0, 25.0, 155.0, 180.0)},
    # Indian Ocean
    {"name": "Indian Ocean Central", "bbox": (-40.0, -5.0, 55.0, 95.0)},
    {"name": "Indian Ocean North", "bbox": (-5.0, 15.0, 55.0, 75.0)},
    # Southern Ocean
    {"name": "Southern Ocean", "bbox": (-75.0, -55.0, -180.0, 180.0)},
    # Arctic
    {"name": "Arctic Ocean", "bbox": (75.0, 90.0, -180.0, 180.0)},
]

# Known large inland water bodies to NOT flag as ocean
# These are rough bounding boxes of major lakes that might be confused
INLAND_WATER_EXCLUSIONS: List[Dict[str, object]] = [
    {"name": "Lake Victoria", "bbox": (-3.1, 0.5, 31.5, 34.9)},
    {"name": "Lake Tanganyika", "bbox": (-8.8, -3.3, 29.0, 31.2)},
    {"name": "Lake Malawi", "bbox": (-14.4, -9.5, 34.0, 35.8)},
    {"name": "Lake Chad", "bbox": (12.0, 14.0, 13.0, 15.5)},
    {"name": "Caspian Sea", "bbox": (36.5, 47.1, 46.6, 54.8)},
    {"name": "Great Lakes NA", "bbox": (41.0, 49.0, -92.0, -76.0)},
]


def _point_in_bbox(
    lat: float, lon: float, bbox: Tuple[float, float, float, float]
) -> bool:
    """Check if a point falls within a bounding box.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        bbox: (lat_min, lat_max, lon_min, lon_max)

    Returns:
        True if the point is within the bounding box.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def is_on_land(lat: float, lon: float) -> bool:
    """Determine if a GPS coordinate is on land (not in ocean).

    Uses simplified ocean region bounding boxes for fast classification.
    This is an approximation -- coastal and island coordinates may be
    incorrectly classified. For production accuracy, use PostGIS with
    coastline polygons.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).

    Returns:
        True if the coordinate is likely on land.
        False if the coordinate is likely in the ocean.

    Note:
        Returns True for coordinates that don't match any known ocean
        region (conservative -- assumes land unless proven ocean).
    """
    # First check if the coordinate is in an inland water exclusion zone
    for exclusion in INLAND_WATER_EXCLUSIONS:
        bbox = exclusion["bbox"]
        if _point_in_bbox(lat, lon, bbox):  # type: ignore[arg-type]
            return True  # In an inland water body area, not ocean

    # Check against major ocean regions
    for region in MAJOR_OCEAN_REGIONS:
        bbox = region["bbox"]
        if _point_in_bbox(lat, lon, bbox):  # type: ignore[arg-type]
            return False  # In a major ocean region

    # Default: assume land (conservative for EUDR validation)
    return True
