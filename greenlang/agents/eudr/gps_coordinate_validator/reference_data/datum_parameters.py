# -*- coding: utf-8 -*-
"""
Datum Transformation Parameters Reference Data - AGENT-EUDR-007

Provides geodetic datum transformation parameters, reference ellipsoid
constants, and country-to-datum mappings for the GPS Coordinate Validator
Agent. All data is deterministic, immutable after module load, and directly
derived from the EPSG Geodetic Parameter Registry (v10.x), IERS
conventions, and NGA Technical Reports.

Datum Parameters:
    46+ geodetic datums with Helmert 7-parameter transformations to WGS84
    (Bursa-Wolf convention: dx, dy, dz translations in metres; rx, ry, rz
    rotations in arc-seconds; ds scale factor in ppm).

Ellipsoid Parameters:
    13 reference ellipsoids with semi-major axis (a), inverse flattening
    (1/f), derived flattening, semi-minor axis (b), and first eccentricity
    squared (e^2).

Country Datum Defaults:
    100+ countries mapped to their most common local geodetic datum,
    enabling automatic datum detection when no explicit datum is declared.

EPSG Sources:
    https://epsg.org/home.html
    EPSG Geodetic Parameter Dataset v10.x
    NGA TR8350.2 (WGS84 Transformations)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference Ellipsoid Parameters
# ---------------------------------------------------------------------------
# Source: EPSG Geodetic Parameter Dataset v10.x
# Each entry: semi_major_axis (a) in metres, inverse_flattening (1/f),
# derived flattening (f), semi_minor_axis (b), eccentricity_squared (e^2),
# EPSG code for the ellipsoid.

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
    "CLARKE_1866": {
        "name": "Clarke 1866",
        "semi_major_axis": 6378206.4,
        "inverse_flattening": 294.9786982,
        "flattening": 1.0 / 294.9786982,
        "semi_minor_axis": 6378206.4 * (1.0 - 1.0 / 294.9786982),
        "eccentricity_squared": 2.0 / 294.9786982 - (1.0 / 294.9786982) ** 2,
        "epsg_code": 7008,
    },
    "CLARKE_1880_IGN": {
        "name": "Clarke 1880 (IGN)",
        "semi_major_axis": 6378249.2,
        "inverse_flattening": 293.466021294,
        "flattening": 1.0 / 293.466021294,
        "semi_minor_axis": 6378249.2 * (1.0 - 1.0 / 293.466021294),
        "eccentricity_squared": 2.0 / 293.466021294 - (1.0 / 293.466021294) ** 2,
        "epsg_code": 7011,
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
        "name": "International 1924 (Hayford)",
        "semi_major_axis": 6378388.0,
        "inverse_flattening": 297.0,
        "flattening": 1.0 / 297.0,
        "semi_minor_axis": 6378388.0 * (1.0 - 1.0 / 297.0),
        "eccentricity_squared": 2.0 / 297.0 - (1.0 / 297.0) ** 2,
        "epsg_code": 7022,
    },
    "KRASSOVSKY_1940": {
        "name": "Krassovsky 1940",
        "semi_major_axis": 6378245.0,
        "inverse_flattening": 298.3,
        "flattening": 1.0 / 298.3,
        "semi_minor_axis": 6378245.0 * (1.0 - 1.0 / 298.3),
        "eccentricity_squared": 2.0 / 298.3 - (1.0 / 298.3) ** 2,
        "epsg_code": 7024,
    },
    "AIRY_1830": {
        "name": "Airy 1830",
        "semi_major_axis": 6377563.396,
        "inverse_flattening": 299.3249646,
        "flattening": 1.0 / 299.3249646,
        "semi_minor_axis": 6377563.396 * (1.0 - 1.0 / 299.3249646),
        "eccentricity_squared": 2.0 / 299.3249646 - (1.0 / 299.3249646) ** 2,
        "epsg_code": 7001,
    },
    "EVEREST_1830": {
        "name": "Everest 1830",
        "semi_major_axis": 6377276.345,
        "inverse_flattening": 300.8017,
        "flattening": 1.0 / 300.8017,
        "semi_minor_axis": 6377276.345 * (1.0 - 1.0 / 300.8017),
        "eccentricity_squared": 2.0 / 300.8017 - (1.0 / 300.8017) ** 2,
        "epsg_code": 7015,
    },
    "FISCHER_1968": {
        "name": "Fischer 1968",
        "semi_major_axis": 6378150.0,
        "inverse_flattening": 298.3,
        "flattening": 1.0 / 298.3,
        "semi_minor_axis": 6378150.0 * (1.0 - 1.0 / 298.3),
        "eccentricity_squared": 2.0 / 298.3 - (1.0 / 298.3) ** 2,
        "epsg_code": 7018,
    },
    "HELMERT_1906": {
        "name": "Helmert 1906",
        "semi_major_axis": 6378200.0,
        "inverse_flattening": 298.3,
        "flattening": 1.0 / 298.3,
        "semi_minor_axis": 6378200.0 * (1.0 - 1.0 / 298.3),
        "eccentricity_squared": 2.0 / 298.3 - (1.0 / 298.3) ** 2,
        "epsg_code": 7020,
    },
    "SOUTH_AMERICAN_1969": {
        "name": "South American 1969",
        "semi_major_axis": 6378160.0,
        "inverse_flattening": 298.25,
        "flattening": 1.0 / 298.25,
        "semi_minor_axis": 6378160.0 * (1.0 - 1.0 / 298.25),
        "eccentricity_squared": 2.0 / 298.25 - (1.0 / 298.25) ** 2,
        "epsg_code": 7036,
    },
}


# ---------------------------------------------------------------------------
# Datum Transformation Parameters (to WGS84)
# ---------------------------------------------------------------------------
# Source: EPSG v10.x, NGA TR8350.2
# Bursa-Wolf convention: dx, dy, dz (metres), rx, ry, rz (arc-seconds),
# ds (ppm).  These are "from datum TO WGS84" transforms.

DATUM_PARAMETERS: Dict[str, Dict[str, Any]] = {
    # ---- WGS84 (identity) ----
    "WGS84": {
        "name": "World Geodetic System 1984",
        "ellipsoid": "WGS84",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Global",
        "accuracy_m": 0.0,
        "countries": [],
    },
    # ---- North America ----
    "NAD27": {
        "name": "North American Datum 1927",
        "ellipsoid": "CLARKE_1866",
        "to_wgs84": {"dx": -8, "dy": 160, "dz": 176, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "North America",
        "accuracy_m": 10.0,
        "countries": ["US", "CA", "MX"],
    },
    "NAD83": {
        "name": "North American Datum 1983",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "North America",
        "accuracy_m": 1.0,
        "countries": ["US", "CA", "MX"],
    },
    # ---- Europe ----
    "ED50": {
        "name": "European Datum 1950",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -87, "dy": -98, "dz": -121, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Europe",
        "accuracy_m": 3.0,
        "countries": ["DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "CH"],
    },
    "ETRS89": {
        "name": "European Terrestrial Reference System 1989",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Europe",
        "accuracy_m": 0.5,
        "countries": ["DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "CZ", "DK", "SE", "FI", "NO"],
    },
    "OSGB_1936": {
        "name": "Ordnance Survey Great Britain 1936",
        "ellipsoid": "AIRY_1830",
        "to_wgs84": {"dx": 446.448, "dy": -125.157, "dz": 542.060, "rx": 0.1502, "ry": 0.2470, "rz": 0.8421, "ds": -20.4894},
        "region": "United Kingdom",
        "accuracy_m": 5.0,
        "countries": ["GB"],
    },
    "EUROPEAN_1979": {
        "name": "European 1979",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -86, "dy": -98, "dz": -119, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Europe",
        "accuracy_m": 3.0,
        "countries": [],
    },
    "POTSDAM": {
        "name": "Potsdam (Rauenberg) Datum",
        "ellipsoid": "BESSEL_1841",
        "to_wgs84": {"dx": 598.1, "dy": 73.7, "dz": 418.2, "rx": 0.202, "ry": 0.045, "rz": -2.455, "ds": 6.7},
        "region": "Germany",
        "accuracy_m": 3.0,
        "countries": ["DE"],
    },
    "ROME_1940": {
        "name": "Rome 1940 (Monte Mario)",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -225, "dy": -65, "dz": 9, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Italy",
        "accuracy_m": 10.0,
        "countries": ["IT"],
    },
    "HERMANNSKOGEL": {
        "name": "Hermannskogel Datum",
        "ellipsoid": "BESSEL_1841",
        "to_wgs84": {"dx": 577.326, "dy": 90.129, "dz": 463.919, "rx": 5.137, "ry": 1.474, "rz": 5.297, "ds": 2.4232},
        "region": "Austria",
        "accuracy_m": 2.0,
        "countries": ["AT"],
    },
    "DATUM_73": {
        "name": "Datum 73 (Portugal)",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -223.237, "dy": 110.193, "dz": 36.649, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Portugal",
        "accuracy_m": 5.0,
        "countries": ["PT"],
    },
    # ---- South America ----
    "SIRGAS_2000": {
        "name": "SIRGAS 2000",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "South America",
        "accuracy_m": 0.5,
        "countries": ["BR", "AR", "CO", "VE", "PE", "EC", "BO", "PY", "UY", "CL", "GY", "SR"],
    },
    "BOGOTA": {
        "name": "Bogota Observatory",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": 307, "dy": 304, "dz": -318, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Colombia",
        "accuracy_m": 15.0,
        "countries": ["CO"],
    },
    "CAMPO_INCHAUSPE": {
        "name": "Campo Inchauspe",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -148, "dy": 136, "dz": 90, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Argentina",
        "accuracy_m": 5.0,
        "countries": ["AR"],
    },
    "CHUA": {
        "name": "Chua Astro",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -134, "dy": 229, "dz": -29, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Paraguay",
        "accuracy_m": 15.0,
        "countries": ["PY"],
    },
    "CORREGO_ALEGRE": {
        "name": "Corrego Alegre 1970-72",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -206.05, "dy": 168.28, "dz": -3.82, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Brazil",
        "accuracy_m": 5.0,
        "countries": ["BR"],
    },
    "HITO_XVIII": {
        "name": "Hito XVIII 1963",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": 16, "dy": 196, "dz": 93, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "South America (Southern Cone)",
        "accuracy_m": 25.0,
        "countries": ["CL", "AR"],
    },
    "LA_CANOA": {
        "name": "La Canoa",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -154, "dy": 29, "dz": -423, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Venezuela",
        "accuracy_m": 15.0,
        "countries": ["VE"],
    },
    "YACARE": {
        "name": "Yacare",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -155, "dy": 171, "dz": 37, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Uruguay",
        "accuracy_m": 10.0,
        "countries": ["UY"],
    },
    # ---- Southeast Asia ----
    "INDIAN_1975": {
        "name": "Indian 1975",
        "ellipsoid": "EVEREST_1830",
        "to_wgs84": {"dx": 210, "dy": 814, "dz": 289, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Thailand",
        "accuracy_m": 12.0,
        "countries": ["TH"],
    },
    "INDIAN_1960": {
        "name": "Indian 1960",
        "ellipsoid": "EVEREST_1830",
        "to_wgs84": {"dx": 198, "dy": 881, "dz": 317, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Vietnam/Cambodia",
        "accuracy_m": 15.0,
        "countries": ["VN", "KH"],
    },
    "INDONESIAN_1974": {
        "name": "Indonesian Datum 1974",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -24, "dy": -15, "dz": 5, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Indonesia",
        "accuracy_m": 10.0,
        "countries": ["ID"],
    },
    "KALIANPUR_1975": {
        "name": "Kalianpur 1975",
        "ellipsoid": "EVEREST_1830",
        "to_wgs84": {"dx": 295, "dy": 736, "dz": 257, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "India",
        "accuracy_m": 10.0,
        "countries": ["IN"],
    },
    "KERTAU_1948": {
        "name": "Kertau 1948",
        "ellipsoid": "EVEREST_1830",
        "to_wgs84": {"dx": -11, "dy": 851, "dz": 5, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "West Malaysia/Singapore",
        "accuracy_m": 10.0,
        "countries": ["MY", "SG", "BN"],
    },
    "LUZON_1911": {
        "name": "Luzon 1911",
        "ellipsoid": "CLARKE_1866",
        "to_wgs84": {"dx": -133, "dy": -77, "dz": -51, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Philippines",
        "accuracy_m": 8.0,
        "countries": ["PH"],
    },
    "TIMBALAI_1948": {
        "name": "Timbalai 1948",
        "ellipsoid": "EVEREST_1830",
        "to_wgs84": {"dx": -679, "dy": 669, "dz": -48, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Brunei/East Malaysia",
        "accuracy_m": 10.0,
        "countries": ["BN", "MY"],
    },
    # ---- Japan / Korea ----
    "TOKYO": {
        "name": "Tokyo Datum",
        "ellipsoid": "BESSEL_1841",
        "to_wgs84": {"dx": -148, "dy": 507, "dz": 685, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Japan/Korea",
        "accuracy_m": 8.0,
        "countries": ["JP", "KR"],
    },
    # ---- Russia / Eastern Europe ----
    "PULKOVO_1942": {
        "name": "Pulkovo 1942",
        "ellipsoid": "KRASSOVSKY_1940",
        "to_wgs84": {"dx": 23.92, "dy": -141.27, "dz": -80.9, "rx": 0, "ry": -0.35, "rz": -0.82, "ds": -0.12},
        "region": "Russia/Eastern Europe",
        "accuracy_m": 5.0,
        "countries": ["RU", "UA", "BY", "KZ", "UZ", "TM", "KG", "TJ", "MN"],
    },
    # ---- Oceania ----
    "GDA94": {
        "name": "Geocentric Datum of Australia 1994",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Australia",
        "accuracy_m": 1.0,
        "countries": ["AU"],
    },
    "GDA2020": {
        "name": "Geocentric Datum of Australia 2020",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Australia",
        "accuracy_m": 0.1,
        "countries": ["AU"],
    },
    "NZGD2000": {
        "name": "New Zealand Geodetic Datum 2000",
        "ellipsoid": "GRS80",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "New Zealand",
        "accuracy_m": 0.1,
        "countries": ["NZ"],
    },
    # ---- Africa ----
    "ARC_1960": {
        "name": "Arc 1960",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -160, "dy": -6, "dz": -302, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "East Africa",
        "accuracy_m": 20.0,
        "countries": ["KE", "TZ", "UG"],
    },
    "ARC_1950": {
        "name": "Arc 1950",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -143, "dy": -90, "dz": -294, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Southern/Central Africa",
        "accuracy_m": 20.0,
        "countries": ["ZW", "ZM", "MW", "BW", "SZ", "LS"],
    },
    "CAPE": {
        "name": "Cape Datum",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -136, "dy": -108, "dz": -292, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "South Africa",
        "accuracy_m": 15.0,
        "countries": ["ZA"],
    },
    "HARTEBEESTHOEK94": {
        "name": "Hartebeesthoek 94",
        "ellipsoid": "WGS84",
        "to_wgs84": {"dx": 0, "dy": 0, "dz": 0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "South Africa",
        "accuracy_m": 1.0,
        "countries": ["ZA"],
    },
    "ADINDAN": {
        "name": "Adindan",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -166, "dy": -15, "dz": 204, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "North/East Africa",
        "accuracy_m": 25.0,
        "countries": ["ET", "SD", "ER"],
    },
    "AFGOOYE": {
        "name": "Afgooye",
        "ellipsoid": "KRASSOVSKY_1940",
        "to_wgs84": {"dx": -43, "dy": -163, "dz": 45, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Somalia",
        "accuracy_m": 25.0,
        "countries": ["SO"],
    },
    "AIN_EL_ABD": {
        "name": "Ain el Abd 1970",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -150, "dy": -251, "dz": -2, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Middle East",
        "accuracy_m": 10.0,
        "countries": ["SA", "BH"],
    },
    "CAMACUPA": {
        "name": "Camacupa",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -50.9, "dy": -347.6, "dz": -231.0, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Angola",
        "accuracy_m": 20.0,
        "countries": ["AO"],
    },
    "CARTHAGE": {
        "name": "Carthage",
        "ellipsoid": "CLARKE_1880_IGN",
        "to_wgs84": {"dx": -263, "dy": 6, "dz": 431, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Tunisia",
        "accuracy_m": 5.0,
        "countries": ["TN"],
    },
    "GANDAJIKA": {
        "name": "Gandajika",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -133, "dy": -321, "dz": 50, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "DR Congo",
        "accuracy_m": 25.0,
        "countries": ["CD"],
    },
    "MINNA": {
        "name": "Minna",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -92, "dy": -93, "dz": 122, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Nigeria",
        "accuracy_m": 15.0,
        "countries": ["NG"],
    },
    "POINT_NOIRE": {
        "name": "Point Noire 1948",
        "ellipsoid": "CLARKE_1880_IGN",
        "to_wgs84": {"dx": -148, "dy": 51, "dz": -291, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Congo",
        "accuracy_m": 25.0,
        "countries": ["CG"],
    },
    "TANANARIVE_1925": {
        "name": "Tananarive 1925",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -189, "dy": -242, "dz": -91, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Madagascar",
        "accuracy_m": 20.0,
        "countries": ["MG"],
    },
    "ZANDERIJ": {
        "name": "Zanderij",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": -265, "dy": 120, "dz": -358, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Suriname",
        "accuracy_m": 15.0,
        "countries": ["SR"],
    },
    # ---- Asia / Middle East ----
    "NAHRWAN": {
        "name": "Nahrwan",
        "ellipsoid": "CLARKE_1880_RGS",
        "to_wgs84": {"dx": -247, "dy": -148, "dz": 369, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Middle East",
        "accuracy_m": 10.0,
        "countries": ["IQ", "OM", "AE"],
    },
    # ---- ISTS ----
    "ISTS_073": {
        "name": "ISTS 073 Astro 1969",
        "ellipsoid": "INTERNATIONAL_1924",
        "to_wgs84": {"dx": 208, "dy": -435, "dz": -229, "rx": 0, "ry": 0, "rz": 0, "ds": 0},
        "region": "Diego Garcia",
        "accuracy_m": 25.0,
        "countries": [],
    },
}


# ---------------------------------------------------------------------------
# Country Datum Defaults
# ---------------------------------------------------------------------------
# ISO 3166-1 alpha-2 -> default local datum key (from DATUM_PARAMETERS)
# Used when coordinate source does not declare a datum.

COUNTRY_DATUM_DEFAULTS: Dict[str, str] = {
    # North America
    "US": "NAD83",
    "CA": "NAD83",
    "MX": "NAD27",
    # South America (EUDR-critical)
    "BR": "SIRGAS_2000",
    "AR": "SIRGAS_2000",
    "CO": "SIRGAS_2000",
    "VE": "SIRGAS_2000",
    "PE": "SIRGAS_2000",
    "EC": "SIRGAS_2000",
    "BO": "SIRGAS_2000",
    "PY": "SIRGAS_2000",
    "UY": "SIRGAS_2000",
    "CL": "SIRGAS_2000",
    "GY": "SIRGAS_2000",
    "SR": "ZANDERIJ",
    "GF": "SIRGAS_2000",
    # Central America
    "GT": "NAD27",
    "HN": "NAD27",
    "NI": "NAD27",
    "CR": "NAD27",
    "PA": "NAD27",
    # Europe
    "DE": "ETRS89",
    "FR": "ETRS89",
    "IT": "ETRS89",
    "ES": "ETRS89",
    "PT": "ETRS89",
    "NL": "ETRS89",
    "BE": "ETRS89",
    "AT": "ETRS89",
    "CH": "ETRS89",
    "PL": "ETRS89",
    "CZ": "ETRS89",
    "SK": "ETRS89",
    "HU": "ETRS89",
    "RO": "ETRS89",
    "BG": "ETRS89",
    "HR": "ETRS89",
    "SI": "ETRS89",
    "SE": "ETRS89",
    "NO": "ETRS89",
    "FI": "ETRS89",
    "DK": "ETRS89",
    "IE": "ETRS89",
    "GB": "OSGB_1936",
    "GR": "ETRS89",
    "EE": "ETRS89",
    "LV": "ETRS89",
    "LT": "ETRS89",
    # Russia / CIS
    "RU": "PULKOVO_1942",
    "UA": "PULKOVO_1942",
    "BY": "PULKOVO_1942",
    "KZ": "PULKOVO_1942",
    "UZ": "PULKOVO_1942",
    "TM": "PULKOVO_1942",
    "KG": "PULKOVO_1942",
    "TJ": "PULKOVO_1942",
    "MN": "PULKOVO_1942",
    # Southeast Asia (EUDR-critical)
    "ID": "INDONESIAN_1974",
    "MY": "KERTAU_1948",
    "TH": "INDIAN_1975",
    "VN": "INDIAN_1960",
    "KH": "INDIAN_1960",
    "MM": "INDIAN_1975",
    "LA": "INDIAN_1960",
    "PH": "LUZON_1911",
    "SG": "KERTAU_1948",
    "BN": "TIMBALAI_1948",
    "IN": "KALIANPUR_1975",
    "LK": "KALIANPUR_1975",
    "PG": "WGS84",
    # Japan / Korea
    "JP": "TOKYO",
    "KR": "TOKYO",
    # Oceania
    "AU": "GDA2020",
    "NZ": "NZGD2000",
    # West Africa (EUDR-critical)
    "GH": "WGS84",
    "CI": "WGS84",
    "NG": "MINNA",
    "CM": "WGS84",
    "TG": "WGS84",
    "GN": "WGS84",
    "SL": "WGS84",
    "LR": "WGS84",
    "SN": "WGS84",
    "BF": "WGS84",
    "BJ": "WGS84",
    "ML": "WGS84",
    "NE": "WGS84",
    # Central Africa (EUDR-critical)
    "CD": "WGS84",
    "CG": "POINT_NOIRE",
    "GA": "WGS84",
    "GQ": "WGS84",
    "CF": "WGS84",
    # East Africa (EUDR-critical)
    "ET": "ADINDAN",
    "KE": "ARC_1960",
    "TZ": "ARC_1960",
    "UG": "ARC_1960",
    "RW": "WGS84",
    "BI": "WGS84",
    "MG": "TANANARIVE_1925",
    "MZ": "WGS84",
    # Southern Africa
    "ZA": "HARTEBEESTHOEK94",
    "ZW": "ARC_1950",
    "ZM": "ARC_1950",
    "MW": "ARC_1950",
    "BW": "ARC_1950",
    "NA": "WGS84",
    "AO": "CAMACUPA",
    # North Africa / Middle East
    "EG": "WGS84",
    "SD": "ADINDAN",
    "ER": "ADINDAN",
    "SO": "AFGOOYE",
    "TN": "CARTHAGE",
    "DZ": "WGS84",
    "MA": "WGS84",
    "LY": "WGS84",
    "SA": "AIN_EL_ABD",
    "IQ": "NAHRWAN",
    "IR": "WGS84",
    "TR": "ED50",
    "AE": "NAHRWAN",
    "OM": "NAHRWAN",
    "BH": "AIN_EL_ABD",
    # China
    "CN": "WGS84",
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_datum_params(datum: str) -> Optional[Dict[str, Any]]:
    """Retrieve datum transformation parameters by datum key.

    Args:
        datum: Datum key string (e.g. 'NAD27', 'ED50', 'WGS84').

    Returns:
        Datum parameter dictionary or None if not found.
    """
    return DATUM_PARAMETERS.get(datum.upper().replace(" ", "_"))


def get_ellipsoid_params(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve reference ellipsoid parameters by name.

    Args:
        name: Ellipsoid name key (e.g. 'WGS84', 'CLARKE_1866').

    Returns:
        Ellipsoid parameter dictionary or None if not found.
    """
    return ELLIPSOID_PARAMETERS.get(name.upper().replace(" ", "_"))


def get_country_default_datum(iso: str) -> str:
    """Retrieve the default local datum for a country.

    Args:
        iso: ISO 3166-1 alpha-2 country code (e.g. 'BR', 'ID').

    Returns:
        Datum key string, defaulting to 'WGS84' if not found.
    """
    return COUNTRY_DATUM_DEFAULTS.get(iso.upper(), "WGS84")


def list_all_datums() -> List[str]:
    """Return a sorted list of all available datum keys.

    Returns:
        Sorted list of datum key strings.
    """
    return sorted(DATUM_PARAMETERS.keys())


def get_transformation_accuracy(source: str, target: str = "WGS84") -> Optional[float]:
    """Estimate transformation accuracy between two datums in metres.

    Currently only supports transformations to WGS84 as target.

    Args:
        source: Source datum key.
        target: Target datum key (default 'WGS84').

    Returns:
        Estimated accuracy in metres, or None if source datum is unknown.
    """
    if target.upper() == "WGS84":
        params = get_datum_params(source)
        if params:
            return params.get("accuracy_m")
    # If both are WGS84-aligned (accuracy ~0), return 0
    source_params = get_datum_params(source)
    target_params = get_datum_params(target)
    if source_params and target_params:
        return (source_params.get("accuracy_m", 0.0)
                + target_params.get("accuracy_m", 0.0))
    return None


# ---------------------------------------------------------------------------
# Module Totals (for __init__.py summary)
# ---------------------------------------------------------------------------

TOTAL_DATUMS: int = len(DATUM_PARAMETERS)
TOTAL_ELLIPSOIDS: int = len(ELLIPSOID_PARAMETERS)
TOTAL_COUNTRY_MAPPINGS: int = len(COUNTRY_DATUM_DEFAULTS)

__all__ = [
    "DATUM_PARAMETERS",
    "ELLIPSOID_PARAMETERS",
    "COUNTRY_DATUM_DEFAULTS",
    "TOTAL_DATUMS",
    "TOTAL_ELLIPSOIDS",
    "TOTAL_COUNTRY_MAPPINGS",
    "get_datum_params",
    "get_ellipsoid_params",
    "get_country_default_datum",
    "list_all_datums",
    "get_transformation_accuracy",
]
