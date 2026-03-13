# -*- coding: utf-8 -*-
"""
Reference Data for Protected Area Validator - AGENT-EUDR-022

Provides authoritative reference data for IUCN categories, protected area
data sources, and country-specific buffer zone regulations.
"""

from greenlang.agents.eudr.protected_area_validator.reference_data.iucn_categories import (
    IUCN_CATEGORY_SCORES,
    IUCN_CATEGORY_DEFINITIONS,
    get_iucn_risk_score,
)
from greenlang.agents.eudr.protected_area_validator.reference_data.protected_area_sources import (
    WDPA_SOURCE,
    NATIONAL_REGISTRIES,
    get_source_metadata,
)
from greenlang.agents.eudr.protected_area_validator.reference_data.country_buffer_regulations import (
    COUNTRY_BUFFER_REGULATIONS,
    get_national_buffer_km,
)

__all__ = [
    "IUCN_CATEGORY_SCORES",
    "IUCN_CATEGORY_DEFINITIONS",
    "get_iucn_risk_score",
    "WDPA_SOURCE",
    "NATIONAL_REGISTRIES",
    "get_source_metadata",
    "COUNTRY_BUFFER_REGULATIONS",
    "get_national_buffer_km",
]
