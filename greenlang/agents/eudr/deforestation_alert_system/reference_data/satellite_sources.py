# -*- coding: utf-8 -*-
"""
Satellite Sources Database - AGENT-EUDR-020 Deforestation Alert System

Comprehensive satellite data source specifications and capabilities for
multi-source deforestation change detection. Covers five satellite platforms
used in the Deforestation Alert System: Sentinel-2 (ESA Copernicus, 10m
multispectral, 5-day revisit), Landsat 8/9 (USGS/NASA, 30m OLI/TIRS,
8-day combined revisit), GLAD (University of Maryland weekly Landsat-based
deforestation alerts), Hansen Global Forest Change (annual Landsat time
series analysis), and RADD (Wageningen University Sentinel-1 SAR-based
radar alerts for detecting deforestation).

Each source entry provides:
    - Platform metadata (operator, launch date, orbit type)
    - Spatial resolution in meters per band group
    - Temporal revisit period (single satellite and constellation)
    - Swath width in kilometers
    - Spectral band configurations with wavelength ranges
    - Coverage by geographic region (tropics, temperate, boreal)
    - Cloud-free effective revisit estimates per climate zone
    - Data availability timelines
    - Applicable spectral vegetation indices

Spectral index formulas:
    - NDVI: (NIR - RED) / (NIR + RED) - vegetation health
    - EVI: 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1) - enhanced
    - NBR: (NIR - SWIR2) / (NIR + SWIR2) - burn severity
    - NDMI: (NIR - SWIR1) / (NIR + SWIR1) - moisture stress
    - SAVI: ((NIR - RED) / (NIR + RED + L)) * (1 + L) - soil adjusted

All numeric values are stored as ``Decimal`` for precision in compliance
calculations and deterministic audit trails.

Data Sources:
    - ESA Copernicus Sentinel-2 Mission Guide v3.0
    - USGS Landsat 8/9 Data Users Handbook v6.0
    - University of Maryland GLAD Forest Alerts Documentation 2024
    - Hansen et al. "High-Resolution Global Maps of 21st-Century Forest
      Cover Change" Science 342 (2013)
    - Reiche et al. "Forest disturbance alerts for the Congo Basin using
      Sentinel-1" Environmental Research Letters 16 (2021)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "ESA Copernicus Sentinel-2 Mission Guide v3.0",
    "USGS Landsat 8/9 Data Users Handbook v6.0",
    "University of Maryland GLAD Forest Alerts Documentation 2024",
    "Hansen et al. Science 342 (2013) Global Forest Change v1.10",
    "Reiche et al. ERL 16 (2021) RADD Forest Disturbance Alerts",
    "ESA Sentinel-1 SAR User Guide v2.0",
]

# ---------------------------------------------------------------------------
# Coverage region constants
# ---------------------------------------------------------------------------

COVERAGE_REGIONS: List[str] = [
    "tropics",
    "subtropics",
    "temperate",
    "boreal",
    "global",
]

# ===========================================================================
# Spectral Index Formulas
# ===========================================================================

SPECTRAL_INDEX_FORMULAS: Dict[str, Dict[str, Any]] = {
    "NDVI": {
        "name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "range_min": Decimal("-1.0"),
        "range_max": Decimal("1.0"),
        "healthy_vegetation_min": Decimal("0.3"),
        "healthy_vegetation_max": Decimal("0.9"),
        "deforestation_threshold": Decimal("-0.15"),
        "description": (
            "Primary vegetation index measuring chlorophyll absorption. "
            "Values above 0.3 indicate healthy vegetation. Sudden drops "
            "below the threshold indicate potential deforestation."
        ),
        "sentinel2_bands": {"NIR": "B8", "RED": "B4"},
        "landsat_bands": {"NIR": "B5", "RED": "B4"},
        "applicable_sources": ["sentinel2", "landsat"],
    },
    "EVI": {
        "name": "Enhanced Vegetation Index",
        "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
        "range_min": Decimal("-1.0"),
        "range_max": Decimal("1.0"),
        "healthy_vegetation_min": Decimal("0.2"),
        "healthy_vegetation_max": Decimal("0.8"),
        "deforestation_threshold": Decimal("-0.12"),
        "description": (
            "Enhanced vegetation index correcting for atmospheric and "
            "canopy background effects. More sensitive than NDVI in high "
            "biomass tropical forests common in EUDR commodity regions."
        ),
        "sentinel2_bands": {"NIR": "B8", "RED": "B4", "BLUE": "B2"},
        "landsat_bands": {"NIR": "B5", "RED": "B4", "BLUE": "B2"},
        "applicable_sources": ["sentinel2", "landsat"],
    },
    "NBR": {
        "name": "Normalized Burn Ratio",
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "range_min": Decimal("-1.0"),
        "range_max": Decimal("1.0"),
        "healthy_vegetation_min": Decimal("0.2"),
        "healthy_vegetation_max": Decimal("0.7"),
        "deforestation_threshold": Decimal("-0.20"),
        "description": (
            "Burn ratio index sensitive to both fire damage and vegetation "
            "moisture content. Useful for detecting slash-and-burn "
            "deforestation common in cattle ranching and smallholder "
            "agriculture expansion."
        ),
        "sentinel2_bands": {"NIR": "B8", "SWIR2": "B12"},
        "landsat_bands": {"NIR": "B5", "SWIR2": "B7"},
        "applicable_sources": ["sentinel2", "landsat"],
    },
    "NDMI": {
        "name": "Normalized Difference Moisture Index",
        "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "range_min": Decimal("-1.0"),
        "range_max": Decimal("1.0"),
        "healthy_vegetation_min": Decimal("0.1"),
        "healthy_vegetation_max": Decimal("0.5"),
        "deforestation_threshold": Decimal("-0.10"),
        "description": (
            "Moisture index measuring vegetation water content. Drops in "
            "NDMI can indicate forest degradation, selective logging, or "
            "early-stage deforestation before complete canopy removal."
        ),
        "sentinel2_bands": {"NIR": "B8", "SWIR1": "B11"},
        "landsat_bands": {"NIR": "B5", "SWIR1": "B6"},
        "applicable_sources": ["sentinel2", "landsat"],
    },
    "SAVI": {
        "name": "Soil-Adjusted Vegetation Index",
        "formula": "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        "range_min": Decimal("-1.0"),
        "range_max": Decimal("1.0"),
        "healthy_vegetation_min": Decimal("0.25"),
        "healthy_vegetation_max": Decimal("0.85"),
        "deforestation_threshold": Decimal("-0.15"),
        "soil_correction_factor_L": Decimal("0.5"),
        "description": (
            "Soil-adjusted index minimizing soil background effects in "
            "areas with sparse vegetation. L factor of 0.5 is standard "
            "for intermediate vegetation density typical of forest edges "
            "and degraded areas."
        ),
        "sentinel2_bands": {"NIR": "B8", "RED": "B4"},
        "landsat_bands": {"NIR": "B5", "RED": "B4"},
        "applicable_sources": ["sentinel2", "landsat"],
    },
}

# ===========================================================================
# Satellite Source Data - 5 satellite platforms
# ===========================================================================

SATELLITE_SOURCE_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # Sentinel-2 (ESA Copernicus)
    # -----------------------------------------------------------------------
    "sentinel2": {
        "name": "Sentinel-2",
        "full_name": "Sentinel-2A/2B Multispectral Instrument",
        "operator": "European Space Agency (ESA)",
        "program": "Copernicus",
        "satellites": ["Sentinel-2A", "Sentinel-2B"],
        "launch_dates": {"2A": "2015-06-23", "2B": "2017-03-07"},
        "orbit_type": "Sun-synchronous",
        "orbit_altitude_km": Decimal("786"),
        "inclination_deg": Decimal("98.62"),
        "resolution_m": 10,
        "resolution_bands": {
            "10m": ["B2", "B3", "B4", "B8"],
            "20m": ["B5", "B6", "B7", "B8A", "B11", "B12"],
            "60m": ["B1", "B9", "B10"],
        },
        "revisit_days_single": 10,
        "revisit_days_constellation": 5,
        "swath_width_km": Decimal("290"),
        "radiometric_resolution_bits": 12,
        "data_format": "JPEG2000",
        "tile_size_km": Decimal("100"),
        "bands": {
            "B1": {
                "name": "Coastal Aerosol",
                "center_wavelength_nm": Decimal("443"),
                "bandwidth_nm": Decimal("20"),
                "resolution_m": 60,
                "purpose": "Aerosol detection and atmospheric correction",
            },
            "B2": {
                "name": "Blue",
                "center_wavelength_nm": Decimal("490"),
                "bandwidth_nm": Decimal("65"),
                "resolution_m": 10,
                "purpose": "Soil/vegetation discrimination, bathymetry",
            },
            "B3": {
                "name": "Green",
                "center_wavelength_nm": Decimal("560"),
                "bandwidth_nm": Decimal("35"),
                "resolution_m": 10,
                "purpose": "Vegetation green peak reflectance",
            },
            "B4": {
                "name": "Red",
                "center_wavelength_nm": Decimal("665"),
                "bandwidth_nm": Decimal("30"),
                "resolution_m": 10,
                "purpose": "Maximum chlorophyll absorption, vegetation indices",
            },
            "B5": {
                "name": "Vegetation Red Edge 1",
                "center_wavelength_nm": Decimal("705"),
                "bandwidth_nm": Decimal("15"),
                "resolution_m": 20,
                "purpose": "Red edge position, chlorophyll content",
            },
            "B6": {
                "name": "Vegetation Red Edge 2",
                "center_wavelength_nm": Decimal("740"),
                "bandwidth_nm": Decimal("15"),
                "resolution_m": 20,
                "purpose": "Red edge inflection, canopy structure",
            },
            "B7": {
                "name": "Vegetation Red Edge 3",
                "center_wavelength_nm": Decimal("783"),
                "bandwidth_nm": Decimal("20"),
                "resolution_m": 20,
                "purpose": "Leaf area index, canopy chlorophyll content",
            },
            "B8": {
                "name": "NIR",
                "center_wavelength_nm": Decimal("842"),
                "bandwidth_nm": Decimal("115"),
                "resolution_m": 10,
                "purpose": "Vegetation health, biomass estimation, NDVI",
            },
            "B8A": {
                "name": "Narrow NIR",
                "center_wavelength_nm": Decimal("865"),
                "bandwidth_nm": Decimal("20"),
                "resolution_m": 20,
                "purpose": "Water vapour correction, vegetation monitoring",
            },
            "B9": {
                "name": "Water Vapour",
                "center_wavelength_nm": Decimal("945"),
                "bandwidth_nm": Decimal("20"),
                "resolution_m": 60,
                "purpose": "Water vapour column estimation",
            },
            "B10": {
                "name": "SWIR - Cirrus",
                "center_wavelength_nm": Decimal("1375"),
                "bandwidth_nm": Decimal("30"),
                "resolution_m": 60,
                "purpose": "Cirrus cloud detection",
            },
            "B11": {
                "name": "SWIR 1",
                "center_wavelength_nm": Decimal("1610"),
                "bandwidth_nm": Decimal("90"),
                "resolution_m": 20,
                "purpose": "Soil/vegetation moisture, NDMI, fire detection",
            },
            "B12": {
                "name": "SWIR 2",
                "center_wavelength_nm": Decimal("2190"),
                "bandwidth_nm": Decimal("180"),
                "resolution_m": 20,
                "purpose": "Geology, soil moisture, NBR, fire severity",
            },
        },
        "coverage": {
            "global": True,
            "tropics": True,
            "subtropics": True,
            "temperate": True,
            "boreal": True,
            "latitude_range_deg": (-56, 84),
        },
        "cloud_free_revisit_days": {
            "tropics": Decimal("25"),
            "subtropics": Decimal("15"),
            "temperate": Decimal("12"),
            "boreal": Decimal("20"),
            "global_average": Decimal("18"),
        },
        "availability_start": "2015-06-23",
        "availability_end": "ongoing",
        "data_access": {
            "primary": "Copernicus Data Space Ecosystem",
            "mirrors": [
                "Google Earth Engine",
                "Amazon Web Services Open Data",
                "Microsoft Planetary Computer",
            ],
            "api": "Copernicus Data Space API (OData + STAC)",
            "license": "Open and free (Copernicus Data Policy)",
        },
        "applicable_indices": ["NDVI", "EVI", "NBR", "NDMI", "SAVI"],
        "eudr_suitability": "excellent",
        "eudr_notes": (
            "Primary optical source for EUDR deforestation monitoring. "
            "10m resolution enables detection of small-scale clearings "
            "in commodity production plots. 5-day revisit provides "
            "near-real-time monitoring capability for tropical regions."
        ),
    },

    # -----------------------------------------------------------------------
    # Landsat 8/9 (USGS/NASA)
    # -----------------------------------------------------------------------
    "landsat": {
        "name": "Landsat 8/9",
        "full_name": "Landsat 8 OLI/TIRS + Landsat 9 OLI-2/TIRS-2",
        "operator": "USGS / NASA",
        "program": "Landsat Continuity Mission",
        "satellites": ["Landsat 8", "Landsat 9"],
        "launch_dates": {"L8": "2013-02-11", "L9": "2021-09-27"},
        "orbit_type": "Sun-synchronous",
        "orbit_altitude_km": Decimal("705"),
        "inclination_deg": Decimal("98.2"),
        "resolution_m": 30,
        "resolution_bands": {
            "15m": ["B8"],
            "30m": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"],
            "100m": ["B10", "B11"],
        },
        "revisit_days_single": 16,
        "revisit_days_constellation": 8,
        "swath_width_km": Decimal("185"),
        "radiometric_resolution_bits": 12,
        "data_format": "GeoTIFF",
        "tile_size_km": Decimal("185"),
        "bands": {
            "B1": {
                "name": "Coastal Aerosol",
                "center_wavelength_nm": Decimal("443"),
                "bandwidth_nm": Decimal("16"),
                "resolution_m": 30,
                "purpose": "Coastal and aerosol studies",
            },
            "B2": {
                "name": "Blue",
                "center_wavelength_nm": Decimal("482"),
                "bandwidth_nm": Decimal("60"),
                "resolution_m": 30,
                "purpose": "Bathymetric mapping, soil/vegetation",
            },
            "B3": {
                "name": "Green",
                "center_wavelength_nm": Decimal("561"),
                "bandwidth_nm": Decimal("57"),
                "resolution_m": 30,
                "purpose": "Vegetation peak reflectance",
            },
            "B4": {
                "name": "Red",
                "center_wavelength_nm": Decimal("655"),
                "bandwidth_nm": Decimal("37"),
                "resolution_m": 30,
                "purpose": "Chlorophyll absorption, vegetation discrimination",
            },
            "B5": {
                "name": "NIR",
                "center_wavelength_nm": Decimal("865"),
                "bandwidth_nm": Decimal("28"),
                "resolution_m": 30,
                "purpose": "Vegetation health, biomass content",
            },
            "B6": {
                "name": "SWIR 1",
                "center_wavelength_nm": Decimal("1609"),
                "bandwidth_nm": Decimal("85"),
                "resolution_m": 30,
                "purpose": "Soil moisture, vegetation water content",
            },
            "B7": {
                "name": "SWIR 2",
                "center_wavelength_nm": Decimal("2201"),
                "bandwidth_nm": Decimal("187"),
                "resolution_m": 30,
                "purpose": "Geology, soil moisture, fire detection",
            },
            "B8": {
                "name": "Panchromatic",
                "center_wavelength_nm": Decimal("590"),
                "bandwidth_nm": Decimal("172"),
                "resolution_m": 15,
                "purpose": "High-resolution visualization, pan-sharpening",
            },
            "B9": {
                "name": "Cirrus",
                "center_wavelength_nm": Decimal("1373"),
                "bandwidth_nm": Decimal("20"),
                "resolution_m": 30,
                "purpose": "Cirrus cloud detection",
            },
            "B10": {
                "name": "TIRS 1",
                "center_wavelength_nm": Decimal("10895"),
                "bandwidth_nm": Decimal("590"),
                "resolution_m": 100,
                "purpose": "Thermal surface temperature mapping",
            },
            "B11": {
                "name": "TIRS 2",
                "center_wavelength_nm": Decimal("12005"),
                "bandwidth_nm": Decimal("1010"),
                "resolution_m": 100,
                "purpose": "Thermal surface temperature mapping",
            },
        },
        "coverage": {
            "global": True,
            "tropics": True,
            "subtropics": True,
            "temperate": True,
            "boreal": True,
            "latitude_range_deg": (-82.7, 82.7),
        },
        "cloud_free_revisit_days": {
            "tropics": Decimal("30"),
            "subtropics": Decimal("20"),
            "temperate": Decimal("16"),
            "boreal": Decimal("25"),
            "global_average": Decimal("23"),
        },
        "availability_start": "1972-07-23",
        "availability_end": "ongoing",
        "data_access": {
            "primary": "USGS EarthExplorer",
            "mirrors": [
                "Google Earth Engine",
                "Amazon Web Services Open Data",
                "Microsoft Planetary Computer",
            ],
            "api": "USGS M2M API",
            "license": "Open and free (USGS Data Policy)",
        },
        "applicable_indices": ["NDVI", "EVI", "NBR", "NDMI", "SAVI"],
        "eudr_suitability": "good",
        "eudr_notes": (
            "Complementary optical source with 50+ year historical archive. "
            "30m resolution suitable for medium-scale deforestation events. "
            "Essential for historical baseline comparison (2018-2020 period) "
            "and pre-cutoff date analysis. Landsat 9 added 2021 for better "
            "temporal coverage."
        ),
    },

    # -----------------------------------------------------------------------
    # GLAD (University of Maryland)
    # -----------------------------------------------------------------------
    "glad": {
        "name": "GLAD",
        "full_name": "Global Land Analysis and Discovery Forest Change Alerts",
        "operator": "University of Maryland / Global Land Analysis & Discovery Lab",
        "program": "GLAD Forest Alerts",
        "satellites": ["Landsat 8", "Landsat 9"],
        "launch_dates": {"alerts_start": "2016-01-01"},
        "orbit_type": "Derived product (Landsat-based)",
        "orbit_altitude_km": Decimal("705"),
        "inclination_deg": Decimal("98.2"),
        "resolution_m": 30,
        "resolution_bands": {
            "30m": ["alert_confidence", "alert_date"],
        },
        "revisit_days_single": 0,
        "revisit_days_constellation": 7,
        "swath_width_km": Decimal("185"),
        "radiometric_resolution_bits": 8,
        "data_format": "GeoTIFF",
        "tile_size_km": Decimal("10"),
        "bands": {
            "alert_confidence": {
                "name": "Alert Confidence",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Confidence level (1=low, 2=nominal, 3=high)",
            },
            "alert_date": {
                "name": "Alert Date",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Julian day of alert detection",
            },
        },
        "coverage": {
            "global": False,
            "tropics": True,
            "subtropics": True,
            "temperate": False,
            "boreal": False,
            "latitude_range_deg": (-30, 30),
        },
        "cloud_free_revisit_days": {
            "tropics": Decimal("7"),
            "subtropics": Decimal("7"),
            "temperate": None,
            "boreal": None,
            "global_average": Decimal("7"),
        },
        "availability_start": "2016-01-01",
        "availability_end": "ongoing",
        "data_access": {
            "primary": "Global Forest Watch",
            "mirrors": [
                "Google Earth Engine",
                "University of Maryland GLAD Lab",
            ],
            "api": "Global Forest Watch API v2",
            "license": "Creative Commons Attribution 4.0",
        },
        "applicable_indices": [],
        "eudr_suitability": "excellent",
        "eudr_notes": (
            "Weekly tropical deforestation alert product specifically "
            "designed for forest change monitoring. Three confidence levels "
            "enable filtering. Essential for near-real-time EUDR monitoring "
            "of tropical commodity source regions (palm oil, soy, cattle, "
            "cocoa, coffee, rubber, wood)."
        ),
    },

    # -----------------------------------------------------------------------
    # Hansen Global Forest Change
    # -----------------------------------------------------------------------
    "hansen_gfc": {
        "name": "Hansen GFC",
        "full_name": "Hansen/UMD/Google/USGS/NASA Global Forest Change",
        "operator": "University of Maryland / Hansen Lab",
        "program": "Global Forest Change",
        "satellites": ["Landsat 5", "Landsat 7", "Landsat 8", "Landsat 9"],
        "launch_dates": {"dataset_start": "2000-01-01"},
        "orbit_type": "Derived product (Landsat time series)",
        "orbit_altitude_km": Decimal("705"),
        "inclination_deg": Decimal("98.2"),
        "resolution_m": 30,
        "resolution_bands": {
            "30m": [
                "treecover2000",
                "loss",
                "gain",
                "lossyear",
                "datamask",
            ],
        },
        "revisit_days_single": 0,
        "revisit_days_constellation": 365,
        "swath_width_km": Decimal("185"),
        "radiometric_resolution_bits": 8,
        "data_format": "GeoTIFF",
        "tile_size_km": Decimal("10"),
        "bands": {
            "treecover2000": {
                "name": "Tree Cover 2000",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Year 2000 tree canopy cover percentage (0-100)",
            },
            "loss": {
                "name": "Forest Loss",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Binary forest loss (1=loss, 0=no loss)",
            },
            "gain": {
                "name": "Forest Gain",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Binary forest gain 2000-2012 (1=gain, 0=no gain)",
            },
            "lossyear": {
                "name": "Loss Year",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Year of forest loss (1-24 for 2001-2024)",
            },
            "datamask": {
                "name": "Data Mask",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 30,
                "purpose": "Land/water/no data mask (0/1/2)",
            },
        },
        "coverage": {
            "global": True,
            "tropics": True,
            "subtropics": True,
            "temperate": True,
            "boreal": True,
            "latitude_range_deg": (-60, 80),
        },
        "cloud_free_revisit_days": {
            "tropics": Decimal("365"),
            "subtropics": Decimal("365"),
            "temperate": Decimal("365"),
            "boreal": Decimal("365"),
            "global_average": Decimal("365"),
        },
        "availability_start": "2000-01-01",
        "availability_end": "ongoing",
        "data_access": {
            "primary": "Global Forest Watch",
            "mirrors": [
                "Google Earth Engine",
                "University of Maryland Hansen Lab",
                "earthenginepartners.appspot.com",
            ],
            "api": "Google Earth Engine Python/JS API",
            "license": "Creative Commons Attribution 4.0",
        },
        "applicable_indices": [],
        "eudr_suitability": "good",
        "eudr_notes": (
            "Annual global tree cover loss product essential for EUDR "
            "cutoff date verification (31 December 2020). The lossyear "
            "layer directly maps to pre/post-cutoff classification. "
            "Global coverage at 30m enables baseline comparison for all "
            "EUDR commodity source countries."
        ),
    },

    # -----------------------------------------------------------------------
    # RADD (Radar Alerts for Detecting Deforestation)
    # -----------------------------------------------------------------------
    "radd": {
        "name": "RADD",
        "full_name": "Radar Alerts for Detecting Deforestation",
        "operator": "Wageningen University & Research",
        "program": "RADD Forest Disturbance Alerts",
        "satellites": ["Sentinel-1A", "Sentinel-1B"],
        "launch_dates": {"1A": "2014-04-03", "1B": "2016-04-25"},
        "orbit_type": "Sun-synchronous (SAR)",
        "orbit_altitude_km": Decimal("693"),
        "inclination_deg": Decimal("98.18"),
        "resolution_m": 10,
        "resolution_bands": {
            "10m": ["VV", "VH", "alert_confidence", "alert_date"],
        },
        "revisit_days_single": 12,
        "revisit_days_constellation": 6,
        "swath_width_km": Decimal("250"),
        "radiometric_resolution_bits": 16,
        "data_format": "GeoTIFF",
        "tile_size_km": Decimal("10"),
        "bands": {
            "VV": {
                "name": "VV Polarization",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 10,
                "purpose": "C-band SAR vertical-vertical backscatter",
            },
            "VH": {
                "name": "VH Polarization",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 10,
                "purpose": "C-band SAR vertical-horizontal backscatter (forest sensitive)",
            },
            "alert_confidence": {
                "name": "Alert Confidence",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 10,
                "purpose": "Alert confidence (2=nominal, 3=high)",
            },
            "alert_date": {
                "name": "Alert Date",
                "center_wavelength_nm": None,
                "bandwidth_nm": None,
                "resolution_m": 10,
                "purpose": "Days since 2018-12-31 of alert detection",
            },
        },
        "coverage": {
            "global": False,
            "tropics": True,
            "subtropics": True,
            "temperate": False,
            "boreal": False,
            "latitude_range_deg": (-30, 30),
        },
        "cloud_free_revisit_days": {
            "tropics": Decimal("6"),
            "subtropics": Decimal("6"),
            "temperate": None,
            "boreal": None,
            "global_average": Decimal("6"),
        },
        "availability_start": "2019-01-01",
        "availability_end": "ongoing",
        "data_access": {
            "primary": "Global Forest Watch",
            "mirrors": [
                "Google Earth Engine",
                "Wageningen University RADD Portal",
            ],
            "api": "Global Forest Watch API v2",
            "license": "Creative Commons Attribution 4.0",
        },
        "applicable_indices": [],
        "eudr_suitability": "excellent",
        "eudr_notes": (
            "SAR-based radar alerts penetrate cloud cover, providing "
            "deforestation detection in persistently cloudy tropical "
            "regions (Congo Basin, Southeast Asia, Amazon wet season). "
            "10m resolution matches Sentinel-2. Critical for filling "
            "optical data gaps during tropical wet seasons when cloud "
            "cover exceeds 80%."
        ),
    },
}


# ===========================================================================
# SatelliteSourceDatabase class
# ===========================================================================


class SatelliteSourceDatabase:
    """
    Stateless reference data accessor for satellite data source specifications.

    Provides typed access to satellite platform specifications, spectral band
    configurations, coverage regions, revisit rates, and availability timelines
    for all 5 satellite sources used in the Deforestation Alert System.

    Example:
        >>> db = SatelliteSourceDatabase()
        >>> specs = db.get_source_specs("sentinel2")
        >>> assert specs["resolution_m"] == 10
        >>> bands = db.get_bands("sentinel2")
        >>> assert "B8" in bands
    """

    def get_source_specs(
        self,
        source_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get full specifications for a satellite source.

        Args:
            source_id: Source identifier (sentinel2, landsat, glad, hansen_gfc, radd).

        Returns:
            Dict with all source specifications, or None if not found.
        """
        return SATELLITE_SOURCE_DATA.get(source_id)

    def get_coverage(
        self,
        source_id: str,
        region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get coverage information for a satellite source.

        Args:
            source_id: Source identifier.
            region: Optional region filter (tropics, subtropics, etc.).

        Returns:
            Coverage dict, or None if source not found.
        """
        source = SATELLITE_SOURCE_DATA.get(source_id)
        if source is None:
            return None
        coverage = source.get("coverage", {})
        if region is not None:
            return {
                "source_id": source_id,
                "region": region,
                "covered": coverage.get(region, False),
            }
        return {
            "source_id": source_id,
            "coverage": coverage,
        }

    def get_bands(
        self,
        source_id: str,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get spectral band configuration for a satellite source.

        Args:
            source_id: Source identifier.

        Returns:
            Dict of band_id -> band specifications, or None if not found.
        """
        source = SATELLITE_SOURCE_DATA.get(source_id)
        if source is None:
            return None
        return source.get("bands", {})

    def get_revisit_rate(
        self,
        source_id: str,
        constellation: bool = True,
    ) -> Optional[int]:
        """Get revisit rate in days for a satellite source.

        Args:
            source_id: Source identifier.
            constellation: If True, return constellation rate (default).

        Returns:
            Revisit period in days, or None if not found.
        """
        source = SATELLITE_SOURCE_DATA.get(source_id)
        if source is None:
            return None
        if constellation:
            return source.get("revisit_days_constellation")
        return source.get("revisit_days_single")

    def get_availability_timeline(
        self,
        source_id: str,
    ) -> Optional[Dict[str, str]]:
        """Get data availability timeline for a satellite source.

        Args:
            source_id: Source identifier.

        Returns:
            Dict with availability_start and availability_end dates.
        """
        source = SATELLITE_SOURCE_DATA.get(source_id)
        if source is None:
            return None
        return {
            "source_id": source_id,
            "availability_start": source.get("availability_start", ""),
            "availability_end": source.get("availability_end", ""),
        }

    def get_source_count(self) -> int:
        """Get total number of satellite sources.

        Returns:
            Number of satellite source entries.
        """
        return len(SATELLITE_SOURCE_DATA)

    def get_all_source_ids(self) -> List[str]:
        """Get list of all satellite source identifiers.

        Returns:
            List of source identifier strings.
        """
        return list(SATELLITE_SOURCE_DATA.keys())

    def get_spectral_index(
        self,
        index_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get spectral index formula and parameters.

        Args:
            index_name: Index name (NDVI, EVI, NBR, NDMI, SAVI).

        Returns:
            Dict with formula, range, thresholds, and band mappings.
        """
        return SPECTRAL_INDEX_FORMULAS.get(index_name)

    def get_cloud_free_revisit(
        self,
        source_id: str,
        region: str = "tropics",
    ) -> Optional[Decimal]:
        """Get effective cloud-free revisit period for a region.

        Args:
            source_id: Source identifier.
            region: Climate region (tropics, subtropics, temperate, boreal).

        Returns:
            Cloud-free revisit period in days, or None.
        """
        source = SATELLITE_SOURCE_DATA.get(source_id)
        if source is None:
            return None
        cloud_free = source.get("cloud_free_revisit_days", {})
        return cloud_free.get(region)

    def get_eudr_suitable_sources(self) -> List[Dict[str, Any]]:
        """Get all sources rated as suitable for EUDR monitoring.

        Returns:
            List of source specs with 'excellent' or 'good' suitability.
        """
        results = []
        for source_id, source in SATELLITE_SOURCE_DATA.items():
            suitability = source.get("eudr_suitability", "")
            if suitability in ("excellent", "good"):
                results.append({
                    "source_id": source_id,
                    "name": source["name"],
                    "resolution_m": source["resolution_m"],
                    "revisit_days": source["revisit_days_constellation"],
                    "eudr_suitability": suitability,
                    "eudr_notes": source.get("eudr_notes", ""),
                })
        return results

    def compare_sources(
        self,
        source_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compare specifications across multiple satellite sources.

        Args:
            source_ids: List of source IDs to compare (default: all).

        Returns:
            List of comparison dicts with key metrics per source.
        """
        ids = source_ids or list(SATELLITE_SOURCE_DATA.keys())
        results = []
        for sid in ids:
            source = SATELLITE_SOURCE_DATA.get(sid)
            if source is None:
                continue
            results.append({
                "source_id": sid,
                "name": source["name"],
                "resolution_m": source["resolution_m"],
                "revisit_days_constellation": source["revisit_days_constellation"],
                "swath_width_km": str(source.get("swath_width_km", "")),
                "band_count": len(source.get("bands", {})),
                "tropics_coverage": source.get("coverage", {}).get("tropics", False),
                "cloud_free_tropics_days": str(
                    source.get("cloud_free_revisit_days", {}).get("tropics", "N/A")
                ),
                "availability_start": source.get("availability_start", ""),
                "eudr_suitability": source.get("eudr_suitability", ""),
            })
        results.sort(key=lambda x: x["resolution_m"])
        return results


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def get_source_specs(source_id: str) -> Optional[Dict[str, Any]]:
    """Get satellite source specifications (module-level convenience).

    Args:
        source_id: Source identifier.

    Returns:
        Source specifications dict or None.
    """
    return SatelliteSourceDatabase().get_source_specs(source_id)


def get_coverage(source_id: str, region: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get satellite source coverage (module-level convenience).

    Args:
        source_id: Source identifier.
        region: Optional region filter.

    Returns:
        Coverage dict or None.
    """
    return SatelliteSourceDatabase().get_coverage(source_id, region)


def get_bands(source_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get satellite source bands (module-level convenience).

    Args:
        source_id: Source identifier.

    Returns:
        Band specifications dict or None.
    """
    return SatelliteSourceDatabase().get_bands(source_id)


def get_revisit_rate(source_id: str, constellation: bool = True) -> Optional[int]:
    """Get satellite revisit rate (module-level convenience).

    Args:
        source_id: Source identifier.
        constellation: If True, return constellation rate.

    Returns:
        Revisit days or None.
    """
    return SatelliteSourceDatabase().get_revisit_rate(source_id, constellation)


def get_availability_timeline(source_id: str) -> Optional[Dict[str, str]]:
    """Get satellite availability timeline (module-level convenience).

    Args:
        source_id: Source identifier.

    Returns:
        Timeline dict or None.
    """
    return SatelliteSourceDatabase().get_availability_timeline(source_id)
