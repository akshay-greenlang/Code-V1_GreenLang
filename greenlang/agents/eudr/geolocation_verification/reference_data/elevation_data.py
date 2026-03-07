# -*- coding: utf-8 -*-
"""
Elevation Reference Data - AGENT-EUDR-002

Provides simplified elevation lookup for coordinate plausibility verification.
Uses regional elevation ranges per EUDR commodity type to flag implausible
coordinates (e.g., cocoa at 5000m elevation).

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Maximum plausible elevation (meters) per commodity type
# Based on known agricultural ranges for EUDR-regulated commodities
COMMODITY_ELEVATION_LIMITS: Dict[str, Tuple[int, int]] = {
    # (min_elevation_m, max_elevation_m)
    "cattle": (-50, 4500),        # Cattle can graze at high altitudes (Andes, Tibet)
    "cocoa": (0, 1200),           # Cocoa grows below ~1200m
    "coffee": (200, 2400),        # Coffee (esp. Arabica) up to ~2400m
    "palm_oil": (-50, 1500),      # Oil palm grows below ~1500m
    "rubber": (0, 1200),          # Rubber trees below ~1200m
    "soya": (-50, 2000),          # Soybean grows up to ~2000m
    "wood": (-50, 4000),          # Timber harvesting up to tree line
}

# Simplified regional elevation grid (10-degree resolution)
# Provides approximate median elevation for broad geographic regions.
# For production use, integrate SRTM or ASTER DEM data.
REGIONAL_ELEVATION_GRID: Dict[Tuple[int, int], int] = {
    # Key: (lat_band_10deg, lon_band_10deg) -> approximate elevation (m)
    # Amazon Basin
    (0, -70): 200, (0, -60): 150, (-10, -70): 300, (-10, -60): 200,
    (-10, -50): 250, (0, -50): 100,
    # Andes
    (-10, -80): 3000, (0, -80): 2500, (-20, -70): 3500, (-30, -70): 2000,
    # Central America
    (10, -90): 500, (10, -80): 300, (20, -100): 1000,
    # West Africa
    (0, 0): 200, (10, 0): 300, (0, 10): 400, (10, -10): 250,
    # Central Africa
    (0, 20): 500, (0, 30): 800, (-10, 20): 400, (-10, 30): 600,
    # East Africa
    (0, 30): 1200, (0, 40): 800, (-10, 30): 1000, (-10, 40): 500,
    # Southeast Asia
    (0, 100): 100, (0, 110): 50, (-10, 110): 100, (0, 120): 200,
    (10, 100): 200, (0, 140): 300, (-10, 140): 200,
    # Europe
    (50, 10): 200, (50, 0): 100, (40, 10): 500, (40, 0): 400,
    # Default for unmatched regions
}


def get_approximate_elevation(lat: float, lon: float) -> Optional[int]:
    """Get approximate elevation for a coordinate using regional grid.

    Uses 10-degree resolution grid. For precise elevation data,
    use SRTM or ASTER DEM integration.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Approximate elevation in meters, or None if no data available.
    """
    lat_band = int(lat // 10) * 10
    lon_band = int(lon // 10) * 10
    return REGIONAL_ELEVATION_GRID.get((lat_band, lon_band))


def is_elevation_plausible(
    lat: float,
    lon: float,
    commodity: str,
    elevation_m: Optional[int] = None,
) -> Tuple[Optional[int], bool]:
    """Check if the elevation at a coordinate is plausible for a commodity.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        commodity: EUDR commodity type (lowercase).
        elevation_m: Known elevation (if available). If None, uses
                     approximate grid lookup.

    Returns:
        Tuple of (elevation_or_None, is_plausible).
    """
    if elevation_m is None:
        elevation_m = get_approximate_elevation(lat, lon)

    if elevation_m is None:
        # No elevation data available -- assume plausible (conservative)
        return None, True

    limits = COMMODITY_ELEVATION_LIMITS.get(commodity.lower())
    if limits is None:
        # Unknown commodity -- assume plausible
        return elevation_m, True

    min_elev, max_elev = limits
    is_ok = min_elev <= elevation_m <= max_elev
    return elevation_m, is_ok
