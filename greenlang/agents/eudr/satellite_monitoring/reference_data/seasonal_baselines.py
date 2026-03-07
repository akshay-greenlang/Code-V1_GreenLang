# -*- coding: utf-8 -*-
"""
Seasonal Baseline Profiles - AGENT-EUDR-003

Provides phenological NDVI adjustment profiles per region for seasonal
normalization. Different biomes in different countries exhibit distinct
seasonal NDVI patterns due to wet/dry seasons, monsoons, and deciduous
leaf cycles. Without seasonal adjustment, natural phenological troughs
can be misclassified as deforestation events (false positives).

Each profile contains 12 monthly NDVI offset values that represent the
expected deviation from the annual mean. These offsets are applied to
raw NDVI observations before change detection to normalize for seasonal
phenology.

Profile range: -0.15 to +0.15 (maximum seasonal NDVI swing)

Data sources:
    - MODIS 16-day NDVI composites (MOD13Q1, 2000-2023)
    - Copernicus Global Land Service NDVI 300m (2017-2023)
    - Peer-reviewed phenology studies per region

EUDR Relevance:
    Seasonal adjustment prevents false positive deforestation alerts
    during natural dry seasons (e.g., Brazilian Cerrado dry season
    June-September, West African harmattan December-February). This
    directly improves the accuracy of Article 9 due diligence
    statements and reduces unnecessary operator investigations.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Seasonal NDVI offset profiles
# ---------------------------------------------------------------------------
#
# Key: (country_code, biome) -> Tuple of 12 monthly NDVI offsets
# Month indices: 0=January, 1=February, ..., 11=December
#
# Positive offset = NDVI expected above annual mean (wet/growing season)
# Negative offset = NDVI expected below annual mean (dry/dormant season)
# Zero offset = no significant seasonal variation

SEASONAL_NDVI_PROFILES: Dict[Tuple[str, str], Tuple[
    float, float, float, float, float, float,
    float, float, float, float, float, float,
]] = {
    # ---- Brazil (BR) ----
    # Amazon rainforest: minimal seasonality, slight dry season Aug-Oct
    ("BR", "tropical_rainforest"): (
        0.03, 0.03, 0.04, 0.04, 0.02, 0.00,
        -0.02, -0.04, -0.05, -0.04, -0.01, 0.00,
    ),
    # Cerrado savanna: strong dry season May-September
    ("BR", "cerrado_savanna"): (
        0.08, 0.10, 0.10, 0.05, -0.02, -0.08,
        -0.12, -0.15, -0.12, -0.02, 0.06, 0.12,
    ),
    # Atlantic Forest: moderate seasonality
    ("BR", "tropical_moist_forest"): (
        0.05, 0.06, 0.06, 0.04, 0.00, -0.04,
        -0.06, -0.08, -0.06, -0.02, 0.02, 0.03,
    ),

    # ---- Indonesia (ID) ----
    # Equatorial rainforest: minimal seasonality, bimodal rainfall
    ("ID", "tropical_rainforest"): (
        0.02, 0.02, 0.03, 0.03, 0.02, 0.00,
        -0.02, -0.03, -0.03, -0.02, 0.00, 0.01,
    ),

    # ---- Malaysia (MY) ----
    # Peninsular/Borneo rainforest: bimodal, NE/SW monsoons
    ("MY", "tropical_rainforest"): (
        0.01, 0.02, 0.03, 0.03, 0.02, 0.00,
        -0.01, -0.02, -0.02, -0.02, -0.02, -0.01,
    ),

    # ---- Colombia (CO) ----
    # Andes cloud forest: bimodal rainfall, two wet seasons
    ("CO", "montane_cloud_forest"): (
        -0.02, -0.03, 0.02, 0.06, 0.08, 0.04,
        0.00, -0.02, -0.04, 0.02, 0.05, -0.01,
    ),
    # Llanos savanna: pronounced wet/dry
    ("CO", "tropical_savanna"): (
        -0.08, -0.10, -0.06, 0.02, 0.08, 0.10,
        0.10, 0.08, 0.04, 0.00, -0.04, -0.06,
    ),
    # Pacific coast moist forest
    ("CO", "tropical_moist_forest"): (
        0.02, 0.03, 0.04, 0.05, 0.03, 0.00,
        -0.02, -0.04, -0.05, -0.03, -0.01, 0.01,
    ),

    # ---- DR Congo (CD) ----
    # Congo basin rainforest: dual rainy seasons
    ("CD", "tropical_rainforest"): (
        0.02, 0.03, 0.04, 0.04, 0.02, -0.01,
        -0.03, -0.04, -0.03, 0.00, 0.02, 0.02,
    ),

    # ---- Republic of Congo (CG) ----
    ("CG", "tropical_rainforest"): (
        0.02, 0.03, 0.04, 0.04, 0.02, -0.01,
        -0.03, -0.04, -0.04, -0.01, 0.01, 0.02,
    ),

    # ---- Peru (PE) ----
    # Amazon lowlands
    ("PE", "tropical_rainforest"): (
        0.04, 0.05, 0.05, 0.03, 0.00, -0.03,
        -0.05, -0.06, -0.05, -0.02, 0.01, 0.03,
    ),
    # Andes cloud forest
    ("PE", "montane_cloud_forest"): (
        0.04, 0.05, 0.06, 0.04, 0.00, -0.04,
        -0.06, -0.07, -0.05, -0.02, 0.02, 0.03,
    ),

    # ---- Ivory Coast (CI) ----
    # West African cocoa belt: bimodal, harmattan Dec-Feb
    ("CI", "tropical_moist_forest"): (
        -0.06, -0.08, -0.04, 0.03, 0.08, 0.10,
        0.08, 0.04, 0.06, 0.05, 0.00, -0.04,
    ),

    # ---- Ghana (GH) ----
    # Similar to CI but slightly shifted
    ("GH", "tropical_moist_forest"): (
        -0.06, -0.07, -0.03, 0.04, 0.08, 0.10,
        0.07, 0.04, 0.05, 0.04, -0.01, -0.05,
    ),

    # ---- Cameroon (CM) ----
    # Equatorial forest zone: bimodal
    ("CM", "tropical_moist_forest"): (
        -0.04, -0.05, -0.02, 0.04, 0.06, 0.08,
        0.06, 0.04, 0.05, 0.03, -0.01, -0.04,
    ),

    # ---- Nigeria (NG) ----
    # Tropical moist forest belt
    ("NG", "tropical_moist_forest"): (
        -0.06, -0.08, -0.04, 0.03, 0.07, 0.10,
        0.08, 0.05, 0.06, 0.04, -0.01, -0.05,
    ),

    # ---- Thailand (TH) ----
    # Monsoon tropical forest: dry Nov-Apr, wet May-Oct
    ("TH", "tropical_moist_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.04, 0.08,
        0.10, 0.10, 0.08, 0.04, -0.02, -0.05,
    ),

    # ---- Vietnam (VN) ----
    # Monsoon pattern: North has cold season, South bimodal
    ("VN", "tropical_moist_forest"): (
        -0.04, -0.05, -0.03, 0.00, 0.04, 0.08,
        0.10, 0.08, 0.06, 0.02, -0.02, -0.04,
    ),

    # ---- Papua New Guinea (PG) ----
    # Equatorial, minimal seasonality
    ("PG", "tropical_rainforest"): (
        0.01, 0.02, 0.02, 0.02, 0.01, 0.00,
        -0.01, -0.02, -0.02, -0.01, 0.00, 0.01,
    ),

    # ---- Bolivia (BO) ----
    # Chiquitano dry forest: strong dry season May-Oct
    ("BO", "tropical_dry_forest"): (
        0.08, 0.10, 0.08, 0.04, -0.02, -0.08,
        -0.10, -0.12, -0.10, -0.04, 0.04, 0.08,
    ),

    # ---- Argentina (AR) ----
    # Chaco: subtropical dry, strong seasonality
    ("AR", "tropical_savanna"): (
        0.10, 0.10, 0.08, 0.02, -0.04, -0.10,
        -0.12, -0.12, -0.08, -0.02, 0.04, 0.08,
    ),

    # ---- Paraguay (PY) ----
    # Chaco/Cerrado transition
    ("PY", "cerrado_savanna"): (
        0.08, 0.10, 0.08, 0.04, -0.02, -0.08,
        -0.12, -0.14, -0.10, -0.02, 0.04, 0.08,
    ),

    # ---- Ethiopia (ET) ----
    # Highland coffee regions: Kiremt rains Jun-Sep
    ("ET", "montane_cloud_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.02, 0.06,
        0.10, 0.12, 0.08, 0.02, -0.04, -0.06,
    ),

    # ---- Ecuador (EC) ----
    # Western slope moist forest
    ("EC", "tropical_moist_forest"): (
        0.04, 0.06, 0.06, 0.04, 0.02, -0.02,
        -0.04, -0.06, -0.06, -0.03, 0.00, 0.02,
    ),

    # ---- Honduras (HN) ----
    # Mesoamerican moist forest
    ("HN", "tropical_moist_forest"): (
        -0.04, -0.06, -0.04, -0.01, 0.04, 0.08,
        0.08, 0.06, 0.06, 0.04, 0.00, -0.03,
    ),
    # Montane coffee zones
    ("HN", "montane_cloud_forest"): (
        -0.05, -0.06, -0.04, 0.00, 0.04, 0.08,
        0.08, 0.06, 0.05, 0.02, -0.02, -0.04,
    ),

    # ---- Guatemala (GT) ----
    ("GT", "montane_cloud_forest"): (
        -0.04, -0.06, -0.04, -0.01, 0.04, 0.08,
        0.08, 0.06, 0.06, 0.03, 0.00, -0.04,
    ),

    # ---- Mexico (MX) ----
    ("MX", "montane_dry_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.02, 0.08,
        0.10, 0.10, 0.08, 0.02, -0.04, -0.06,
    ),

    # ---- Myanmar (MM) ----
    ("MM", "tropical_moist_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.04, 0.08,
        0.10, 0.10, 0.08, 0.04, -0.02, -0.05,
    ),

    # ---- Cambodia (KH) ----
    ("KH", "tropical_moist_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.04, 0.08,
        0.10, 0.10, 0.08, 0.04, -0.02, -0.05,
    ),

    # ---- Uganda (UG) ----
    ("UG", "tropical_moist_forest"): (
        0.00, -0.02, 0.04, 0.06, 0.06, 0.02,
        -0.02, -0.04, -0.02, 0.02, 0.04, 0.00,
    ),

    # ---- Kenya (KE) ----
    ("KE", "montane_cloud_forest"): (
        -0.06, -0.08, -0.02, 0.06, 0.08, 0.04,
        0.00, -0.02, -0.04, 0.02, 0.06, -0.02,
    ),

    # ---- Tanzania (TZ) ----
    ("TZ", "montane_cloud_forest"): (
        0.02, 0.00, 0.04, 0.08, 0.06, -0.02,
        -0.06, -0.08, -0.06, -0.02, 0.02, 0.04,
    ),

    # ---- Gabon (GA) ----
    ("GA", "tropical_rainforest"): (
        0.02, 0.03, 0.04, 0.04, 0.02, -0.02,
        -0.04, -0.04, -0.03, 0.00, 0.01, 0.02,
    ),

    # ---- Liberia (LR) ----
    ("LR", "tropical_moist_forest"): (
        -0.06, -0.07, -0.03, 0.03, 0.08, 0.10,
        0.08, 0.04, 0.05, 0.04, -0.01, -0.05,
    ),

    # ---- India (IN) ----
    # Western Ghats / Kerala rubber belt: SW monsoon Jun-Sep
    ("IN", "tropical_moist_forest"): (
        -0.04, -0.06, -0.04, -0.02, 0.02, 0.08,
        0.10, 0.10, 0.06, 0.02, -0.02, -0.04,
    ),

    # ---- Rwanda (RW) ----
    ("RW", "montane_cloud_forest"): (
        -0.02, -0.04, 0.02, 0.06, 0.06, 0.02,
        -0.02, -0.04, -0.04, 0.02, 0.04, 0.00,
    ),

    # ---- Laos (LA) ----
    ("LA", "tropical_moist_forest"): (
        -0.06, -0.08, -0.06, -0.02, 0.04, 0.08,
        0.10, 0.10, 0.08, 0.04, -0.02, -0.05,
    ),

    # ---- Sri Lanka (LK) ----
    ("LK", "tropical_moist_forest"): (
        -0.02, -0.03, -0.01, 0.02, 0.06, 0.06,
        0.04, 0.02, 0.02, 0.04, 0.02, -0.02,
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_seasonal_adjustment(
    country_code: str,
    biome: str,
    month: int,
) -> float:
    """Get the seasonal NDVI adjustment for a specific month and region.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (uppercase).
        biome: Biome name (e.g., 'tropical_rainforest').
        month: Month number (1-12), where 1=January, 12=December.

    Returns:
        NDVI offset value for the specified month. Returns 0.0 if no
        seasonal profile is available for the country/biome combination.

    Raises:
        ValueError: If month is not in range [1, 12].
    """
    if not (1 <= month <= 12):
        raise ValueError(
            f"month must be in [1, 12], got {month}"
        )

    profile = SEASONAL_NDVI_PROFILES.get(
        (country_code.upper(), biome)
    )
    if profile is None:
        return 0.0

    # month is 1-indexed, tuple is 0-indexed
    return profile[month - 1]


def adjust_ndvi_for_season(
    ndvi: float,
    country_code: str,
    biome: str,
    observation_date: Union[date, datetime],
) -> float:
    """Adjust an NDVI observation for seasonal phenology.

    Subtracts the expected seasonal offset from the observed NDVI to
    produce a seasonally-normalized value. This normalization reduces
    false positive change detections caused by natural phenological
    cycles.

    The adjustment formula is::

        adjusted_ndvi = observed_ndvi - seasonal_offset

    During the dry season (negative offset), the adjustment increases
    the NDVI, compensating for the expected natural decrease. During
    the wet season (positive offset), the adjustment decreases the
    NDVI, compensating for the expected natural increase.

    Args:
        ndvi: Observed NDVI value in range [-1.0, 1.0].
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome name for phenological profile selection.
        observation_date: Date of the NDVI observation.

    Returns:
        Seasonally-adjusted NDVI value, clamped to [-1.0, 1.0].
    """
    month = observation_date.month
    offset = get_seasonal_adjustment(country_code, biome, month)

    adjusted = ndvi - offset

    # Clamp to valid NDVI range
    return max(-1.0, min(1.0, adjusted))


def get_seasonal_profile(
    country_code: str,
    biome: str,
) -> Optional[Tuple[
    float, float, float, float, float, float,
    float, float, float, float, float, float,
]]:
    """Retrieve the full 12-month seasonal NDVI profile for a region.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome name.

    Returns:
        Tuple of 12 monthly NDVI offset values (Jan-Dec), or None
        if no profile exists for the country/biome combination.
    """
    return SEASONAL_NDVI_PROFILES.get(
        (country_code.upper(), biome)
    )


def get_peak_green_month(
    country_code: str,
    biome: str,
) -> Optional[int]:
    """Determine the month with peak greenness for a region.

    Useful for selecting optimal baseline imagery windows when the
    maximum NDVI contrast between forest and non-forest is expected.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome name.

    Returns:
        Month number (1-12) with the highest NDVI offset, or None if
        no seasonal profile is available.
    """
    profile = get_seasonal_profile(country_code, biome)
    if profile is None:
        return None

    max_offset = max(profile)
    # Return 1-indexed month
    return profile.index(max_offset) + 1


def get_dry_season_months(
    country_code: str,
    biome: str,
    threshold: float = -0.03,
) -> list[int]:
    """Identify the months considered part of the dry season.

    Dry season months are those with NDVI offsets below the given
    threshold, indicating significantly reduced greenness relative
    to the annual mean.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome name.
        threshold: NDVI offset below which a month is considered
            dry season. Defaults to -0.03.

    Returns:
        List of month numbers (1-12) in the dry season. Empty list
        if no seasonal profile is available or no months qualify.
    """
    profile = get_seasonal_profile(country_code, biome)
    if profile is None:
        return []

    return [
        month_idx + 1
        for month_idx, offset in enumerate(profile)
        if offset < threshold
    ]


def has_seasonal_profile(
    country_code: str,
    biome: str,
) -> bool:
    """Check if a seasonal NDVI profile exists for a region.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome name.

    Returns:
        True if a seasonal profile is available.
    """
    return (country_code.upper(), biome) in SEASONAL_NDVI_PROFILES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SEASONAL_NDVI_PROFILES",
    "get_seasonal_adjustment",
    "adjust_ndvi_for_season",
    "get_seasonal_profile",
    "get_peak_green_month",
    "get_dry_season_months",
    "has_seasonal_profile",
]
