# -*- coding: utf-8 -*-
"""
GreenLang Data Module

This module provides emission factor management, unit normalization,
GWP timeframe handling, and heating value conversion utilities.

ZERO-HALLUCINATION GUARANTEE:
- All data from authoritative sources (IPCC, EPA, DEFRA)
- Deterministic calculations (same input -> same output)
- Full provenance tracking
- No LLM in calculation path
"""

from greenlang.data.emission_factors import EmissionFactors
from greenlang.data.heating_value_converter import (
    HeatingValueBasis,
    HeatingValueConverter,
    hhv_to_lhv,
    lhv_to_hhv,
    get_hhv_lhv_ratio,
)
from greenlang.data.unit_normalizer import (
    UnitNormalizer,
    normalize_unit,
    units_are_equivalent,
    normalize_unit_for_storage,
)
from greenlang.data.gwp_timeframes import (
    GWPAssessmentReport,
    GWPTimeframe,
    GWPReference,
    GWPRegistry,
    get_gwp,
    get_regulatory_default,
)

__all__ = [
    # Emission Factors
    "EmissionFactors",

    # HHV/LHV Conversion
    "HeatingValueBasis",
    "HeatingValueConverter",
    "hhv_to_lhv",
    "lhv_to_hhv",
    "get_hhv_lhv_ratio",

    # Unit Normalization (NEW - fixes kWh vs kwh bug)
    "UnitNormalizer",
    "normalize_unit",
    "units_are_equivalent",
    "normalize_unit_for_storage",

    # GWP Timeframes (NEW - AR5, SAR support)
    "GWPAssessmentReport",
    "GWPTimeframe",
    "GWPReference",
    "GWPRegistry",
    "get_gwp",
    "get_regulatory_default",
]
