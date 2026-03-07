# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-004: Forest Cover Analysis Agent

Provides built-in reference datasets for forest cover analysis operations:
    - biome_parameters: Biome-specific spectral and structural parameters
    - allometric_equations: Biomass allometric models and SAR calibration
    - forest_definitions: FAO/EUDR forest definitions and classification criteria

These datasets enable deterministic, zero-hallucination forest analysis
without external API dependencies. All data is version-tracked and
provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
"""

from greenlang.agents.eudr.forest_cover_analysis.reference_data.biome_parameters import (
    BIOME_LIST,
    BIOME_PARAMETERS,
    COMMODITY_BIOME_MAP,
    get_biome_params,
)
from greenlang.agents.eudr.forest_cover_analysis.reference_data.allometric_equations import (
    ALLOMETRIC_EQUATIONS,
    NDVI_REGRESSION_COEFFICIENTS,
    SAR_COEFFICIENTS,
    get_allometric_equation,
)
from greenlang.agents.eudr.forest_cover_analysis.reference_data.forest_definitions import (
    EUDR_FOREST_EXCLUSIONS,
    FAO_FOREST_DEFINITION,
    FOREST_TYPE_CRITERIA,
    get_forest_definition,
)

__all__ = [
    "BIOME_PARAMETERS",
    "get_biome_params",
    "BIOME_LIST",
    "COMMODITY_BIOME_MAP",
    "ALLOMETRIC_EQUATIONS",
    "get_allometric_equation",
    "SAR_COEFFICIENTS",
    "NDVI_REGRESSION_COEFFICIENTS",
    "FAO_FOREST_DEFINITION",
    "EUDR_FOREST_EXCLUSIONS",
    "FOREST_TYPE_CRITERIA",
    "get_forest_definition",
]
