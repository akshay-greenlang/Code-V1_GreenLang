# -*- coding: utf-8 -*-
"""
Satellite Sensor Specifications - AGENT-EUDR-003

Provides band-level specifications for all satellite sensors supported by
the Satellite Monitoring Agent. Each band entry includes central wavelength,
spatial resolution, and a human-readable description.

Supported sensors:
    - Sentinel-2 MSI (13 bands, 10m/20m/60m, ESA Copernicus)
    - Landsat 8 OLI/TIRS (11 bands, 15m/30m/100m, USGS)
    - Landsat 9 OLI-2/TIRS-2 (11 bands, 15m/30m/100m, USGS)
    - Sentinel-1 SAR (C-band, IW/EW/SM modes, ESA Copernicus)

These specifications are used by:
    - ImageryAcquisitionEngine: Band selection for download
    - SpectralIndexCalculator: Band mapping for index formulas
    - DataFusionEngine: Resolution harmonization across sensors
    - CloudGapFiller: SAR band selection for cloud-free alternatives

Data sources:
    - ESA Sentinel-2 User Handbook (2024)
    - USGS Landsat 8-9 Data Users Handbook (2024)
    - ESA Sentinel-1 User Guide (2024)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sentinel-2 MSI band specifications
# ---------------------------------------------------------------------------
#
# Key: band name (e.g., "B01", "B02", ..., "B12", "B8A")
# Value: (central_wavelength_nm, spatial_resolution_m, description)

SENTINEL2_BANDS: Dict[str, Tuple[int, int, str]] = {
    "B01": (443, 60, "Coastal aerosol"),
    "B02": (490, 10, "Blue"),
    "B03": (560, 10, "Green"),
    "B04": (665, 10, "Red"),
    "B05": (705, 20, "Vegetation red edge 1"),
    "B06": (740, 20, "Vegetation red edge 2"),
    "B07": (783, 20, "Vegetation red edge 3"),
    "B08": (842, 10, "NIR"),
    "B8A": (865, 20, "Narrow NIR"),
    "B09": (945, 60, "Water vapour"),
    "B10": (1375, 60, "SWIR - Cirrus"),
    "B11": (1610, 20, "SWIR 1"),
    "B12": (2190, 20, "SWIR 2"),
}

# ---------------------------------------------------------------------------
# Landsat 8 OLI/TIRS band specifications
# ---------------------------------------------------------------------------
#
# Key: band name (e.g., "B1", "B2", ..., "B11")
# Value: (central_wavelength_nm, spatial_resolution_m, description)

LANDSAT8_BANDS: Dict[str, Tuple[int, int, str]] = {
    "B1": (443, 30, "Coastal aerosol"),
    "B2": (482, 30, "Blue"),
    "B3": (562, 30, "Green"),
    "B4": (655, 30, "Red"),
    "B5": (865, 30, "NIR"),
    "B6": (1609, 30, "SWIR 1"),
    "B7": (2201, 30, "SWIR 2"),
    "B8": (590, 15, "Panchromatic"),
    "B9": (1373, 30, "Cirrus"),
    "B10": (10895, 100, "Thermal infrared 1"),
    "B11": (12005, 100, "Thermal infrared 2"),
}

# ---------------------------------------------------------------------------
# Landsat 9 OLI-2/TIRS-2 band specifications
# ---------------------------------------------------------------------------
#
# Landsat 9 is functionally identical to Landsat 8 in band layout.
# The OLI-2 and TIRS-2 instruments have improved radiometric resolution
# (14-bit vs 12-bit quantization) and improved thermal band performance.

LANDSAT9_BANDS: Dict[str, Tuple[int, int, str]] = {
    "B1": (443, 30, "Coastal aerosol"),
    "B2": (482, 30, "Blue"),
    "B3": (562, 30, "Green"),
    "B4": (655, 30, "Red"),
    "B5": (865, 30, "NIR"),
    "B6": (1609, 30, "SWIR 1"),
    "B7": (2201, 30, "SWIR 2"),
    "B8": (590, 15, "Panchromatic"),
    "B9": (1373, 30, "Cirrus"),
    "B10": (10895, 100, "Thermal infrared 1"),
    "B11": (12005, 100, "Thermal infrared 2"),
}

# ---------------------------------------------------------------------------
# Sentinel-1 SAR mode specifications
# ---------------------------------------------------------------------------
#
# Key: mode abbreviation
# Value: dict with mode properties

SENTINEL1_MODES: Dict[str, Dict[str, object]] = {
    "IW": {
        "name": "Interferometric Wide Swath",
        "swath_width_km": 250,
        "resolution_m": 5,
        "pixel_spacing_m": 10,
        "polarizations": ["VV", "VH", "VV+VH"],
        "frequency_ghz": 5.405,
        "band": "C",
        "description": (
            "Default mode over land. 250 km swath with 5x20m resolution "
            "in single-look. Most commonly used for deforestation monitoring."
        ),
    },
    "EW": {
        "name": "Extra Wide Swath",
        "swath_width_km": 400,
        "resolution_m": 20,
        "pixel_spacing_m": 40,
        "polarizations": ["HH", "HV", "HH+HV"],
        "frequency_ghz": 5.405,
        "band": "C",
        "description": (
            "Wide-area mode primarily for maritime and polar monitoring. "
            "400 km swath with reduced 20x40m resolution."
        ),
    },
    "SM": {
        "name": "Stripmap",
        "swath_width_km": 80,
        "resolution_m": 5,
        "pixel_spacing_m": 5,
        "polarizations": ["HH", "VV", "HH+HV", "VV+VH"],
        "frequency_ghz": 5.405,
        "band": "C",
        "description": (
            "High-resolution mode for targeted monitoring. 80 km swath "
            "with 5x5m resolution. Used for detailed site investigations."
        ),
    },
}

# ---------------------------------------------------------------------------
# Spectral index band requirements
# ---------------------------------------------------------------------------
#
# Maps spectral index names to the bands required for their computation
# across different satellite sensors.

_INDEX_BAND_MAP: Dict[str, Dict[str, List[str]]] = {
    "NDVI": {
        "sentinel2": ["B04", "B08"],       # Red, NIR
        "landsat8": ["B4", "B5"],           # Red, NIR
        "landsat9": ["B4", "B5"],           # Red, NIR
    },
    "EVI": {
        "sentinel2": ["B02", "B04", "B08"],  # Blue, Red, NIR
        "landsat8": ["B2", "B4", "B5"],      # Blue, Red, NIR
        "landsat9": ["B2", "B4", "B5"],      # Blue, Red, NIR
    },
    "NBR": {
        "sentinel2": ["B08", "B12"],       # NIR, SWIR2
        "landsat8": ["B5", "B7"],           # NIR, SWIR2
        "landsat9": ["B5", "B7"],           # NIR, SWIR2
    },
    "NDMI": {
        "sentinel2": ["B08", "B11"],       # NIR, SWIR1
        "landsat8": ["B5", "B6"],           # NIR, SWIR1
        "landsat9": ["B5", "B6"],           # NIR, SWIR1
    },
    "SAVI": {
        "sentinel2": ["B04", "B08"],       # Red, NIR
        "landsat8": ["B4", "B5"],           # Red, NIR
        "landsat9": ["B4", "B5"],           # Red, NIR
    },
    "NDWI": {
        "sentinel2": ["B03", "B08"],       # Green, NIR
        "landsat8": ["B3", "B5"],           # Green, NIR
        "landsat9": ["B3", "B5"],           # Green, NIR
    },
    "BSI": {
        "sentinel2": ["B02", "B04", "B08", "B11"],  # Blue, Red, NIR, SWIR1
        "landsat8": ["B2", "B4", "B5", "B6"],        # Blue, Red, NIR, SWIR1
        "landsat9": ["B2", "B4", "B5", "B6"],        # Blue, Red, NIR, SWIR1
    },
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_band_for_index(
    source: str,
    index_type: str,
) -> List[str]:
    """Get the band names required to compute a spectral index.

    Args:
        source: Satellite source identifier. One of 'sentinel2',
            'landsat8', or 'landsat9'.
        index_type: Spectral index name (case-insensitive). One of
            'NDVI', 'EVI', 'NBR', 'NDMI', 'SAVI', 'NDWI', or 'BSI'.

    Returns:
        List of band name strings required for the index computation.
        Returns an empty list if the source or index is not recognized.
    """
    index_entry = _INDEX_BAND_MAP.get(index_type.upper())
    if index_entry is None:
        return []

    return list(index_entry.get(source.lower(), []))


def get_resolution(
    source: str,
    band: str,
) -> Optional[int]:
    """Get the spatial resolution (in meters) for a specific band.

    Args:
        source: Satellite source identifier. One of 'sentinel2',
            'landsat8', or 'landsat9'.
        band: Band name (e.g., 'B04' for Sentinel-2, 'B5' for Landsat).

    Returns:
        Resolution in meters, or None if the source/band is not
        recognized.
    """
    band_specs = _get_band_specs(source)
    if band_specs is None:
        return None

    entry = band_specs.get(band)
    if entry is None:
        return None

    return entry[1]


def get_wavelength(
    source: str,
    band: str,
) -> Optional[int]:
    """Get the central wavelength (in nanometers) for a specific band.

    Args:
        source: Satellite source identifier. One of 'sentinel2',
            'landsat8', or 'landsat9'.
        band: Band name.

    Returns:
        Central wavelength in nanometers, or None if the source/band
        is not recognized.
    """
    band_specs = _get_band_specs(source)
    if band_specs is None:
        return None

    entry = band_specs.get(band)
    if entry is None:
        return None

    return entry[0]


def get_band_description(
    source: str,
    band: str,
) -> Optional[str]:
    """Get the human-readable description for a specific band.

    Args:
        source: Satellite source identifier.
        band: Band name.

    Returns:
        Band description string, or None if not recognized.
    """
    band_specs = _get_band_specs(source)
    if band_specs is None:
        return None

    entry = band_specs.get(band)
    if entry is None:
        return None

    return entry[2]


def list_bands(source: str) -> List[str]:
    """List all available band names for a satellite source.

    Args:
        source: Satellite source identifier. One of 'sentinel2',
            'landsat8', or 'landsat9'.

    Returns:
        Sorted list of band name strings. Empty list if the source
        is not recognized.
    """
    band_specs = _get_band_specs(source)
    if band_specs is None:
        return []
    return sorted(band_specs.keys())


def list_supported_indices() -> List[str]:
    """List all spectral indices supported by the system.

    Returns:
        Sorted list of spectral index names.
    """
    return sorted(_INDEX_BAND_MAP.keys())


def get_native_resolution(source: str) -> Optional[int]:
    """Get the best (finest) native resolution for a satellite source.

    Returns the smallest resolution value across all bands for the
    given source. Useful for determining the finest detail available.

    Args:
        source: Satellite source identifier.

    Returns:
        Finest resolution in meters, or None if source is not recognized.
    """
    band_specs = _get_band_specs(source)
    if band_specs is None:
        return None

    resolutions = [entry[1] for entry in band_specs.values()]
    return min(resolutions) if resolutions else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_band_specs(
    source: str,
) -> Optional[Dict[str, Tuple[int, int, str]]]:
    """Return the band specification dictionary for a satellite source.

    Args:
        source: Satellite source identifier (case-insensitive).

    Returns:
        Band specification dictionary, or None if not recognized.
    """
    source_lower = source.lower()
    if source_lower == "sentinel2":
        return SENTINEL2_BANDS
    if source_lower == "landsat8":
        return LANDSAT8_BANDS
    if source_lower == "landsat9":
        return LANDSAT9_BANDS
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SENTINEL2_BANDS",
    "LANDSAT8_BANDS",
    "LANDSAT9_BANDS",
    "SENTINEL1_MODES",
    "get_band_for_index",
    "get_resolution",
    "get_wavelength",
    "get_band_description",
    "list_bands",
    "list_supported_indices",
    "get_native_resolution",
]
