# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-007: GPS Coordinate Validator Agent

Provides built-in reference datasets for GPS coordinate validation:
    - datum_parameters: 46+ geodetic datums, 13 ellipsoids, 100+ country defaults
    - country_boundaries: 200+ country bounding boxes, centroids, ocean regions
    - commodity_zones: 7 EUDR commodity growing zones, 500+ cities for urban detection

These datasets enable deterministic, zero-hallucination coordinate validation,
datum transformation, plausibility checking, and compliance assessment without
external API dependencies. All data is version-tracked and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from greenlang.agents.eudr.gps_coordinate_validator.reference_data.datum_parameters import (
    COUNTRY_DATUM_DEFAULTS,
    DATUM_PARAMETERS,
    ELLIPSOID_PARAMETERS,
    TOTAL_COUNTRY_MAPPINGS,
    TOTAL_DATUMS,
    TOTAL_ELLIPSOIDS,
    get_country_default_datum,
    get_datum_params,
    get_ellipsoid_params,
    get_transformation_accuracy,
    list_all_datums,
)
from greenlang.agents.eudr.gps_coordinate_validator.reference_data.country_boundaries import (
    COUNTRY_BOUNDARIES,
    OCEAN_REGIONS,
    TOTAL_COUNTRIES,
    TOTAL_OCEAN_REGIONS,
    find_country,
    get_country,
    get_country_centroid,
    get_eudr_countries,
    is_ocean,
)
from greenlang.agents.eudr.gps_coordinate_validator.reference_data.commodity_zones import (
    COMMODITY_ZONES,
    MAJOR_CITIES,
    TOTAL_CITIES,
    TOTAL_COMMODITIES,
    get_commodity_zones,
    get_elevation_range,
    get_major_producers,
    is_commodity_plausible,
    is_urban,
)

__all__ = [
    # datum_parameters
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
    # country_boundaries
    "COUNTRY_BOUNDARIES",
    "OCEAN_REGIONS",
    "TOTAL_COUNTRIES",
    "TOTAL_OCEAN_REGIONS",
    "get_country",
    "find_country",
    "is_ocean",
    "get_eudr_countries",
    "get_country_centroid",
    # commodity_zones
    "COMMODITY_ZONES",
    "MAJOR_CITIES",
    "TOTAL_COMMODITIES",
    "TOTAL_CITIES",
    "is_commodity_plausible",
    "get_commodity_zones",
    "is_urban",
    "get_elevation_range",
    "get_major_producers",
]
