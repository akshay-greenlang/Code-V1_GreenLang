# -*- coding: utf-8 -*-
"""
Projection Parameters Reference Data - AGENT-EUDR-006

Provides CRS definitions, UTM zone parameters, datum transformation parameters,
and reference ellipsoid constants for the Plot Boundary Manager Agent. All data
is deterministic, immutable after module load, and directly derived from the
EPSG Geodetic Parameter Registry (v10.x) and IERS conventions.

CRS Definitions:
    50+ EPSG codes covering WGS84, UTM North/South zones, Web Mercator,
    SIRGAS 2000, ETRS89, NAD83, NZGD2000, GDA94, and GDA2020.

UTM Zone Parameters:
    All 120 UTM zones (60 North + 60 South) with central meridian,
    scale factor, false easting, and false northing.

Transformation Parameters:
    7-parameter Helmert transformations for NAD83->WGS84, ETRS89->WGS84,
    SIRGAS 2000->WGS84, GDA94->WGS84, and GDA2020->WGS84.

Ellipsoid Parameters:
    WGS84, GRS80, Clarke 1880, Bessel 1841, and International 1924
    reference ellipsoids with semi-major axis and inverse flattening.

EPSG Sources:
    https://epsg.org/home.html
    EPSG Geodetic Parameter Dataset v10.x

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference ellipsoid parameters
# ---------------------------------------------------------------------------
# Source: EPSG Geodetic Parameter Dataset v10.x
# Each entry: semi_major_axis (a) in metres, inverse_flattening (1/f)

ELLIPSOID_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "WGS84": {
        "name": "World Geodetic System 1984",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "flattening": 1.0 / 298.257223563,
        "semi_minor_axis": 6378137.0 * (1.0 - 1.0 / 298.257223563),
        "eccentricity_squared": 2.0 / 298.257223563 - (1.0 / 298.257223563) ** 2,
        "epsg_code": 7030,
    },
    "GRS80": {
        "name": "Geodetic Reference System 1980",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257222101,
        "flattening": 1.0 / 298.257222101,
        "semi_minor_axis": 6378137.0 * (1.0 - 1.0 / 298.257222101),
        "eccentricity_squared": 2.0 / 298.257222101 - (1.0 / 298.257222101) ** 2,
        "epsg_code": 7019,
    },
    "CLARKE_1880_RGS": {
        "name": "Clarke 1880 (RGS)",
        "semi_major_axis": 6378249.145,
        "inverse_flattening": 293.465,
        "flattening": 1.0 / 293.465,
        "semi_minor_axis": 6378249.145 * (1.0 - 1.0 / 293.465),
        "eccentricity_squared": 2.0 / 293.465 - (1.0 / 293.465) ** 2,
        "epsg_code": 7012,
    },
    "BESSEL_1841": {
        "name": "Bessel 1841",
        "semi_major_axis": 6377397.155,
        "inverse_flattening": 299.1528128,
        "flattening": 1.0 / 299.1528128,
        "semi_minor_axis": 6377397.155 * (1.0 - 1.0 / 299.1528128),
        "eccentricity_squared": 2.0 / 299.1528128 - (1.0 / 299.1528128) ** 2,
        "epsg_code": 7004,
    },
    "INTERNATIONAL_1924": {
        "name": "International 1924",
        "semi_major_axis": 6378388.0,
        "inverse_flattening": 297.0,
        "flattening": 1.0 / 297.0,
        "semi_minor_axis": 6378388.0 * (1.0 - 1.0 / 297.0),
        "eccentricity_squared": 2.0 / 297.0 - (1.0 / 297.0) ** 2,
        "epsg_code": 7022,
    },
}

# ---------------------------------------------------------------------------
# CRS definitions (50+ EPSG codes)
# ---------------------------------------------------------------------------
# Each entry: name, type (geographic/projected), datum, ellipsoid, units, bounds
# Bounds are [west, south, east, north] in degrees (for geographic) or metres
# (for projected).

CRS_DEFINITIONS: Dict[int, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # Global geographic CRS
    # -----------------------------------------------------------------------
    4326: {
        "name": "WGS 84",
        "type": "geographic",
        "datum": "World Geodetic System 1984",
        "ellipsoid": "WGS84",
        "units": "degrees",
        "bounds": [-180.0, -90.0, 180.0, 90.0],
        "authority": "EPSG",
        "area_of_use": "World",
    },
    3857: {
        "name": "WGS 84 / Pseudo-Mercator",
        "type": "projected",
        "datum": "World Geodetic System 1984",
        "ellipsoid": "WGS84",
        "units": "metres",
        "bounds": [-20037508.34, -20048966.10, 20037508.34, 20048966.10],
        "authority": "EPSG",
        "area_of_use": "World between 85.06S and 85.06N",
    },

    # -----------------------------------------------------------------------
    # South American regional CRS
    # -----------------------------------------------------------------------
    4674: {
        "name": "SIRGAS 2000",
        "type": "geographic",
        "datum": "Sistema de Referencia Geocentrico para las Americas 2000",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [-122.19, -59.87, -25.28, 32.72],
        "authority": "EPSG",
        "area_of_use": "Latin America",
    },

    # -----------------------------------------------------------------------
    # European regional CRS
    # -----------------------------------------------------------------------
    4258: {
        "name": "ETRS89",
        "type": "geographic",
        "datum": "European Terrestrial Reference System 1989",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [-16.1, 32.88, 40.18, 84.73],
        "authority": "EPSG",
        "area_of_use": "Europe",
    },
    3035: {
        "name": "ETRS89 / LAEA Europe",
        "type": "projected",
        "datum": "European Terrestrial Reference System 1989",
        "ellipsoid": "GRS80",
        "units": "metres",
        "bounds": [1896628.62, 1095703.18, 6293974.43, 5765143.74],
        "authority": "EPSG",
        "area_of_use": "Europe",
    },

    # -----------------------------------------------------------------------
    # North American regional CRS
    # -----------------------------------------------------------------------
    4269: {
        "name": "NAD83",
        "type": "geographic",
        "datum": "North American Datum 1983",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [-172.54, 14.92, -47.74, 86.46],
        "authority": "EPSG",
        "area_of_use": "North America",
    },

    # -----------------------------------------------------------------------
    # New Zealand
    # -----------------------------------------------------------------------
    4167: {
        "name": "NZGD2000",
        "type": "geographic",
        "datum": "New Zealand Geodetic Datum 2000",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [160.6, -55.95, -171.2, -25.88],
        "authority": "EPSG",
        "area_of_use": "New Zealand",
    },

    # -----------------------------------------------------------------------
    # Australia
    # -----------------------------------------------------------------------
    4283: {
        "name": "GDA94",
        "type": "geographic",
        "datum": "Geocentric Datum of Australia 1994",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [93.41, -60.55, 173.34, -8.47],
        "authority": "EPSG",
        "area_of_use": "Australia",
    },
    7844: {
        "name": "GDA2020",
        "type": "geographic",
        "datum": "Geocentric Datum of Australia 2020",
        "ellipsoid": "GRS80",
        "units": "degrees",
        "bounds": [93.41, -60.55, 173.34, -8.47],
        "authority": "EPSG",
        "area_of_use": "Australia",
    },
}

# ---------------------------------------------------------------------------
# UTM North zones (EPSG:32601 - EPSG:32660)
# ---------------------------------------------------------------------------

for _zone in range(1, 61):
    _epsg = 32600 + _zone
    _cm = -183.0 + _zone * 6.0
    _west = _cm - 3.0
    _east = _cm + 3.0
    CRS_DEFINITIONS[_epsg] = {
        "name": f"WGS 84 / UTM zone {_zone}N",
        "type": "projected",
        "datum": "World Geodetic System 1984",
        "ellipsoid": "WGS84",
        "units": "metres",
        "bounds": [166021.44, 0.0, 833978.56, 9329005.18],
        "authority": "EPSG",
        "area_of_use": f"Northern Hemisphere, {_west:.0f}E to {_east:.0f}E",
        "utm_zone": _zone,
        "utm_hemisphere": "N",
        "central_meridian": _cm,
    }

# ---------------------------------------------------------------------------
# UTM South zones (EPSG:32701 - EPSG:32760)
# ---------------------------------------------------------------------------

for _zone in range(1, 61):
    _epsg = 32700 + _zone
    _cm = -183.0 + _zone * 6.0
    _west = _cm - 3.0
    _east = _cm + 3.0
    CRS_DEFINITIONS[_epsg] = {
        "name": f"WGS 84 / UTM zone {_zone}S",
        "type": "projected",
        "datum": "World Geodetic System 1984",
        "ellipsoid": "WGS84",
        "units": "metres",
        "bounds": [166021.44, 1116915.04, 833978.56, 10000000.0],
        "authority": "EPSG",
        "area_of_use": f"Southern Hemisphere, {_west:.0f}E to {_east:.0f}E",
        "utm_zone": _zone,
        "utm_hemisphere": "S",
        "central_meridian": _cm,
    }

# ---------------------------------------------------------------------------
# ETRS89 / UTM zones (commonly used in Europe)
# ---------------------------------------------------------------------------

for _zone_num, _etrs_epsg in [
    (28, 25828), (29, 25829), (30, 25830), (31, 25831),
    (32, 25832), (33, 25833), (34, 25834), (35, 25835),
    (36, 25836), (37, 25837), (38, 25838),
]:
    _cm = -183.0 + _zone_num * 6.0
    CRS_DEFINITIONS[_etrs_epsg] = {
        "name": f"ETRS89 / UTM zone {_zone_num}N",
        "type": "projected",
        "datum": "European Terrestrial Reference System 1989",
        "ellipsoid": "GRS80",
        "units": "metres",
        "bounds": [166021.44, 0.0, 833978.56, 9329005.18],
        "authority": "EPSG",
        "area_of_use": f"Europe, UTM zone {_zone_num}N",
        "utm_zone": _zone_num,
        "utm_hemisphere": "N",
        "central_meridian": _cm,
    }

# ---------------------------------------------------------------------------
# SIRGAS 2000 / UTM zones (commonly used in Brazil)
# ---------------------------------------------------------------------------

for _zone_num, _sirgas_epsg in [
    (21, 31981), (22, 31982), (23, 31983), (24, 31984), (25, 31985),
]:
    _cm = -183.0 + _zone_num * 6.0
    CRS_DEFINITIONS[_sirgas_epsg] = {
        "name": f"SIRGAS 2000 / UTM zone {_zone_num}S",
        "type": "projected",
        "datum": "Sistema de Referencia Geocentrico para las Americas 2000",
        "ellipsoid": "GRS80",
        "units": "metres",
        "bounds": [166021.44, 1116915.04, 833978.56, 10000000.0],
        "authority": "EPSG",
        "area_of_use": f"South America, UTM zone {_zone_num}S",
        "utm_zone": _zone_num,
        "utm_hemisphere": "S",
        "central_meridian": _cm,
    }

# ---------------------------------------------------------------------------
# GDA2020 / MGA zones (Australia Map Grid of Australia)
# ---------------------------------------------------------------------------

for _zone_num, _mga_epsg in [
    (49, 7849), (50, 7850), (51, 7851), (52, 7852),
    (53, 7853), (54, 7854), (55, 7855), (56, 7856),
]:
    _cm = -183.0 + _zone_num * 6.0
    CRS_DEFINITIONS[_mga_epsg] = {
        "name": f"GDA2020 / MGA zone {_zone_num}",
        "type": "projected",
        "datum": "Geocentric Datum of Australia 2020",
        "ellipsoid": "GRS80",
        "units": "metres",
        "bounds": [166021.44, 1116915.04, 833978.56, 10000000.0],
        "authority": "EPSG",
        "area_of_use": f"Australia, MGA zone {_zone_num}",
        "utm_zone": _zone_num,
        "utm_hemisphere": "S",
        "central_meridian": _cm,
    }

# Clean up loop variables from module namespace
del _zone, _epsg, _cm, _west, _east, _zone_num
try:
    del _etrs_epsg, _sirgas_epsg, _mga_epsg
except NameError:
    pass

# ---------------------------------------------------------------------------
# UTM zone parameters (all 120 zones)
# ---------------------------------------------------------------------------
# Parameters per EPSG Guidance Note 7-2 (Universal Transverse Mercator)
# Scale factor: 0.9996
# False easting: 500000 metres
# False northing: 0 (North) / 10000000 (South)

UTM_ZONE_PARAMETERS: Dict[str, Dict[str, Any]] = {}

for _z in range(1, 61):
    _central_meridian = -183.0 + _z * 6.0

    # North hemisphere
    UTM_ZONE_PARAMETERS[f"{_z}N"] = {
        "zone_number": _z,
        "hemisphere": "N",
        "central_meridian": _central_meridian,
        "scale_factor": 0.9996,
        "false_easting": 500000.0,
        "false_northing": 0.0,
        "epsg_code": 32600 + _z,
        "latitude_of_origin": 0.0,
        "longitude_range": (_central_meridian - 3.0, _central_meridian + 3.0),
    }

    # South hemisphere
    UTM_ZONE_PARAMETERS[f"{_z}S"] = {
        "zone_number": _z,
        "hemisphere": "S",
        "central_meridian": _central_meridian,
        "scale_factor": 0.9996,
        "false_easting": 500000.0,
        "false_northing": 10000000.0,
        "epsg_code": 32700 + _z,
        "latitude_of_origin": 0.0,
        "longitude_range": (_central_meridian - 3.0, _central_meridian + 3.0),
    }

del _z, _central_meridian

# ---------------------------------------------------------------------------
# Datum transformation parameters (7-parameter Helmert)
# ---------------------------------------------------------------------------
# Source: EPSG Geodetic Parameter Dataset, coordinate operation methods
#
# Parameters: dx, dy, dz (metres), rx, ry, rz (arc-seconds), ds (ppm)
# Sign convention: Position Vector (EPSG method 1033)
#
# These enable approximate transformations. For high-accuracy work,
# use grid-based transformations (NTv2/NADCON) when available.

TRANSFORMATION_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "NAD83_TO_WGS84": {
        "source_crs": "NAD83",
        "source_epsg": 4269,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.9956,
        "dy": -1.9013,
        "dz": -0.5215,
        "rx": 0.025915,
        "ry": 0.009426,
        "rz": 0.011599,
        "ds": 0.00062,
        "accuracy_metres": 1.0,
        "epsg_operation": 15851,
        "note": (
            "NAD83 to WGS84 (CORS96). For practical purposes "
            "NAD83 and WGS84 are coincident at ~1m accuracy."
        ),
    },
    "ETRS89_TO_WGS84": {
        "source_crs": "ETRS89",
        "source_epsg": 4258,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "ds": 0.0,
        "accuracy_metres": 1.0,
        "epsg_operation": 1149,
        "note": (
            "ETRS89 to WGS84 identity at epoch 1989.0. "
            "Divergence grows ~2.5cm/year due to plate tectonics."
        ),
    },
    "SIRGAS2000_TO_WGS84": {
        "source_crs": "SIRGAS 2000",
        "source_epsg": 4674,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "ds": 0.0,
        "accuracy_metres": 1.0,
        "epsg_operation": 4075,
        "note": (
            "SIRGAS 2000 to WGS84 identity. SIRGAS 2000 is a "
            "realisation of ITRS at epoch 2000.4, practically "
            "identical to WGS84 (G1150)."
        ),
    },
    "GDA94_TO_WGS84": {
        "source_crs": "GDA94",
        "source_epsg": 4283,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.06155,
        "dy": -0.01087,
        "dz": -0.04019,
        "rx": -0.0394924,
        "ry": -0.0327221,
        "rz": -0.0328979,
        "ds": -0.009994,
        "accuracy_metres": 0.05,
        "epsg_operation": 15931,
        "note": (
            "GDA94 to WGS84 via ITRF2008. ICSM published "
            "parameters."
        ),
    },
    "GDA2020_TO_WGS84": {
        "source_crs": "GDA2020",
        "source_epsg": 7844,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "ds": 0.0,
        "accuracy_metres": 0.03,
        "epsg_operation": 8447,
        "note": (
            "GDA2020 to WGS84 (G1762) identity at epoch 2020.0. "
            "GDA2020 is aligned to ITRF2014 at epoch 2020.0."
        ),
    },
    "NZGD2000_TO_WGS84": {
        "source_crs": "NZGD2000",
        "source_epsg": 4167,
        "target_crs": "WGS84",
        "target_epsg": 4326,
        "method": "Position Vector 7-parameter",
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "ds": 0.0,
        "accuracy_metres": 1.0,
        "epsg_operation": 1565,
        "note": (
            "NZGD2000 to WGS84 identity. NZGD2000 is based on "
            "ITRF96 at epoch 2000.0."
        ),
    },
    "GDA94_TO_GDA2020": {
        "source_crs": "GDA94",
        "source_epsg": 4283,
        "target_crs": "GDA2020",
        "target_epsg": 7844,
        "method": "Position Vector 7-parameter",
        "dx": -0.06155,
        "dy": 0.01087,
        "dz": 0.04019,
        "rx": 0.0394924,
        "ry": 0.0327221,
        "rz": 0.0328979,
        "ds": 0.009994,
        "accuracy_metres": 0.05,
        "epsg_operation": 8048,
        "note": (
            "GDA94 to GDA2020 conformal 7-parameter. Inverse "
            "of GDA94->WGS84 with GDA2020 treated as WGS84-aligned."
        ),
    },
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_crs_definition(epsg_code: int) -> Optional[Dict[str, Any]]:
    """Return the CRS definition for a given EPSG code.

    Args:
        epsg_code: EPSG code to look up (e.g. 4326, 32748).

    Returns:
        Dictionary with CRS properties, or None if not found.

    Example:
        >>> defn = get_crs_definition(4326)
        >>> defn["name"]
        'WGS 84'
        >>> defn["type"]
        'geographic'
    """
    return CRS_DEFINITIONS.get(epsg_code)


def get_utm_zone(longitude: float) -> int:
    """Calculate the UTM zone number from a longitude value.

    Uses the standard formula: zone = floor((longitude + 180) / 6) + 1,
    clamped to the range [1, 60].

    Special cases for Svalbard and Norway are NOT handled here; use the
    full MGRS/UTM library for those edge cases.

    Args:
        longitude: Longitude in decimal degrees, range [-180, 180].

    Returns:
        UTM zone number in the range [1, 60].

    Example:
        >>> get_utm_zone(0.0)
        31
        >>> get_utm_zone(-73.0)
        18
        >>> get_utm_zone(103.8)
        48
    """
    zone = int(math.floor((longitude + 180.0) / 6.0)) + 1
    return max(1, min(60, zone))


def get_utm_hemisphere(latitude: float) -> str:
    """Determine the UTM hemisphere from a latitude value.

    Args:
        latitude: Latitude in decimal degrees, range [-90, 90].

    Returns:
        'N' for Northern Hemisphere (latitude >= 0),
        'S' for Southern Hemisphere (latitude < 0).

    Example:
        >>> get_utm_hemisphere(5.0)
        'N'
        >>> get_utm_hemisphere(-6.2)
        'S'
    """
    return "N" if latitude >= 0.0 else "S"


def get_utm_epsg(longitude: float, latitude: float) -> int:
    """Return the EPSG code for the UTM zone covering the given coordinate.

    Combines ``get_utm_zone`` and ``get_utm_hemisphere`` to produce the
    full EPSG code.

    Args:
        longitude: Longitude in decimal degrees.
        latitude: Latitude in decimal degrees.

    Returns:
        EPSG code (326xx for North, 327xx for South).

    Example:
        >>> get_utm_epsg(-73.0, 5.0)
        32618
        >>> get_utm_epsg(103.8, -6.2)
        32748
    """
    zone = get_utm_zone(longitude)
    if latitude >= 0.0:
        return 32600 + zone
    return 32700 + zone


def get_utm_zone_parameters(
    zone_number: int,
    hemisphere: str,
) -> Optional[Dict[str, Any]]:
    """Return UTM zone parameters for a specific zone and hemisphere.

    Args:
        zone_number: UTM zone number in [1, 60].
        hemisphere: 'N' for North, 'S' for South.

    Returns:
        Dictionary with zone parameters, or None if invalid zone.

    Example:
        >>> params = get_utm_zone_parameters(48, 'S')
        >>> params["central_meridian"]
        105.0
        >>> params["false_northing"]
        10000000.0
    """
    key = f"{zone_number}{hemisphere.upper()}"
    return UTM_ZONE_PARAMETERS.get(key)


def get_transformation(
    source_epsg: int,
    target_epsg: int,
) -> Optional[Dict[str, Any]]:
    """Find a datum transformation between two CRS codes.

    Searches the TRANSFORMATION_PARAMETERS dictionary for a matching
    source/target EPSG pair.

    Args:
        source_epsg: EPSG code of the source CRS.
        target_epsg: EPSG code of the target CRS.

    Returns:
        Transformation parameter dictionary, or None if no
        transformation is defined.

    Example:
        >>> t = get_transformation(4269, 4326)
        >>> t["method"]
        'Position Vector 7-parameter'
    """
    for params in TRANSFORMATION_PARAMETERS.values():
        if (
            params["source_epsg"] == source_epsg
            and params["target_epsg"] == target_epsg
        ):
            return params
    return None


def get_ellipsoid(name: str) -> Optional[Dict[str, Any]]:
    """Return ellipsoid parameters by name.

    Args:
        name: Ellipsoid name (e.g. 'WGS84', 'GRS80', 'BESSEL_1841').

    Returns:
        Dictionary with ellipsoid parameters, or None if not found.

    Example:
        >>> e = get_ellipsoid("WGS84")
        >>> e["semi_major_axis"]
        6378137.0
    """
    return ELLIPSOID_PARAMETERS.get(name.upper().replace(" ", "_"))


def get_central_meridian(longitude: float) -> float:
    """Return the central meridian for the UTM zone covering the longitude.

    Args:
        longitude: Longitude in decimal degrees.

    Returns:
        Central meridian in decimal degrees.

    Example:
        >>> get_central_meridian(0.0)
        3.0
        >>> get_central_meridian(-73.0)
        -75.0
    """
    zone = get_utm_zone(longitude)
    return -183.0 + zone * 6.0


def is_geographic_crs(epsg_code: int) -> bool:
    """Check whether a CRS is geographic (lat/lon) rather than projected.

    Args:
        epsg_code: EPSG code to check.

    Returns:
        True if the CRS is geographic, False otherwise (or if unknown).

    Example:
        >>> is_geographic_crs(4326)
        True
        >>> is_geographic_crs(32748)
        False
    """
    defn = CRS_DEFINITIONS.get(epsg_code)
    if defn is None:
        return False
    return defn.get("type") == "geographic"


def is_projected_crs(epsg_code: int) -> bool:
    """Check whether a CRS is projected (easting/northing).

    Args:
        epsg_code: EPSG code to check.

    Returns:
        True if the CRS is projected, False otherwise (or if unknown).

    Example:
        >>> is_projected_crs(32748)
        True
        >>> is_projected_crs(4326)
        False
    """
    defn = CRS_DEFINITIONS.get(epsg_code)
    if defn is None:
        return False
    return defn.get("type") == "projected"


def get_all_epsg_codes() -> Tuple[int, ...]:
    """Return a sorted tuple of all defined EPSG codes.

    Returns:
        Tuple of EPSG codes in ascending order.
    """
    return tuple(sorted(CRS_DEFINITIONS.keys()))


def get_all_utm_zone_keys() -> Tuple[str, ...]:
    """Return a sorted tuple of all UTM zone keys (e.g. '1N', '1S', ...).

    Returns:
        Tuple of zone keys in natural sort order.
    """
    return tuple(sorted(
        UTM_ZONE_PARAMETERS.keys(),
        key=lambda k: (int(k[:-1]), k[-1]),
    ))


# ---------------------------------------------------------------------------
# Module-level total CRS count for introspection
# ---------------------------------------------------------------------------

TOTAL_CRS_COUNT: int = len(CRS_DEFINITIONS)
TOTAL_UTM_ZONES: int = len(UTM_ZONE_PARAMETERS)
TOTAL_TRANSFORMATIONS: int = len(TRANSFORMATION_PARAMETERS)
TOTAL_ELLIPSOIDS: int = len(ELLIPSOID_PARAMETERS)
