# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-002: Geolocation Verification Agent

Provides built-in reference datasets for geolocation verification:
    - country_boundaries: Country bounding boxes for 250+ countries
    - ocean_mask: Land/ocean classification
    - protected_areas: Major WDPA/Ramsar/UNESCO protected areas
    - elevation_data: Simplified elevation lookup

These datasets enable offline verification without external API dependencies.
All data is deterministic and version-tracked.

Author: GreenLang Platform Team
Date: March 2026
"""

from greenlang.agents.eudr.geolocation_verification.reference_data.country_boundaries import (
    COUNTRY_BOUNDING_BOXES,
    get_country_for_coordinate,
    is_coordinate_in_country,
)
from greenlang.agents.eudr.geolocation_verification.reference_data.ocean_mask import (
    is_on_land,
    MAJOR_OCEAN_REGIONS,
)

__all__ = [
    "COUNTRY_BOUNDING_BOXES",
    "get_country_for_coordinate",
    "is_coordinate_in_country",
    "is_on_land",
    "MAJOR_OCEAN_REGIONS",
]
