# -*- coding: utf-8 -*-
"""
Shared Climate & Weather Data Types
====================================

Cross-cutting enumerations and base models used by both the Weather &
Climate Data Connector (GL-DATA-X-008) and the Climate Hazard Connector
(AGENT-DATA-020 / GL-DATA-GEO-002).

Extracted types:
    - GeoCoordinate: WGS84 coordinate base model (latitude, longitude,
      elevation) shared by both Location models.
    - ClimateScenario: Unified IPCC SSP (CMIP6) and RCP (CMIP5) scenario
      pathways used by both weather projections and hazard scenario analysis.
    - TimeHorizon: IPCC AR6-aligned time horizon windows for climate
      projections.

Design notes:
    - Both agents previously defined independent Location models and
      scenario enums with slightly different field sets and value formats.
    - This module provides the canonical definitions so both agents share
      a single source of truth for geographic coordinates and climate
      scenario identifiers.
    - Existing Location classes in each agent remain intact and compose
      the GeoCoordinate base with their own additional fields.

Author: GreenLang Team
Version: 1.0.0
"""

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator

from greenlang.schemas import GreenLangBase


# =============================================================================
# BASE MODELS
# =============================================================================

class GeoCoordinate(GreenLangBase):
    """WGS84 geographic coordinate with optional elevation.

    Provides the minimal geographic point representation shared across
    all climate and weather data models. Both the Weather Climate Agent's
    ``Location`` and the Climate Hazard Connector's ``Location`` extend
    this base with their own domain-specific fields.

    Attributes:
        latitude: WGS84 latitude in decimal degrees (-90 to 90).
        longitude: WGS84 longitude in decimal degrees (-180 to 180).
        elevation_m: Elevation above mean sea level in metres (optional).
    """

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees (-90 to 90)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees (-180 to 180)",
    )
    elevation_m: Optional[float] = Field(
        None,
        description="Elevation above mean sea level in metres",
    )

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within WGS84 bounds."""
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"latitude must be between -90 and 90, got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within WGS84 bounds."""
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"longitude must be between -180 and 180, got {v}")
        return v


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ClimateScenario(str, Enum):
    """Unified IPCC climate projection scenario pathways.

    Covers both CMIP6 Shared Socioeconomic Pathways (SSP) and legacy
    CMIP5 Representative Concentration Pathways (RCP). This canonical
    enum is the single source of truth for scenario identifiers across
    the Weather Climate Agent and the Climate Hazard Connector.

    Value format uses dots for SSP (e.g. ``ssp1_1.9``) to remain
    backward-compatible with the Weather Climate Agent's original
    values and underscores for RCP (e.g. ``rcp2.6``).

    SSP1_19: Sustainability; very low emissions; 1.5C target.
    SSP1_26: Sustainability; low emissions; well-below 2C.
    SSP2_45: Middle of the road; intermediate emissions.
    SSP3_70: Regional rivalry; high emissions.
    SSP5_85: Fossil-fuelled development; very high emissions.
    RCP26: Legacy CMIP5 low emissions pathway (approx. 2C).
    RCP45: Legacy CMIP5 intermediate emissions pathway.
    RCP85: Legacy CMIP5 high emissions pathway (business-as-usual).
    """

    SSP1_19 = "ssp1_1.9"
    SSP1_26 = "ssp1_2.6"
    SSP2_45 = "ssp2_4.5"
    SSP3_70 = "ssp3_7.0"
    SSP5_85 = "ssp5_8.5"
    RCP26 = "rcp2.6"
    RCP45 = "rcp4.5"
    RCP85 = "rcp8.5"


class TimeHorizon(str, Enum):
    """Climate projection time horizons aligned with IPCC AR6 conventions.

    Defines the temporal windows used for climate hazard projections
    and scenario analysis. Each horizon maps to a specific year range
    used for data aggregation and risk scoring.

    BASELINE: 1995-2014; IPCC AR6 reference period for historical data.
    NEAR_TERM: 2021-2040; short-range projection window.
    MID_TERM: 2041-2060; medium-range projection window.
    LONG_TERM: 2061-2080; long-range projection window.
    END_CENTURY: 2081-2100; end-of-century projection window.
    """

    BASELINE = "baseline"
    NEAR_TERM = "near_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    END_CENTURY = "end_century"


__all__ = [
    "GeoCoordinate",
    "ClimateScenario",
    "TimeHorizon",
]
