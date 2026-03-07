# -*- coding: utf-8 -*-
"""
Allometric Equations and SAR Calibration - AGENT-EUDR-004

Provides biomass allometric models, SAR backscatter calibration coefficients,
NDVI regression parameters, and height allometric equations for all 16 biome
types. These enable deterministic (zero-hallucination) biomass, height, and
canopy density estimation from remote sensing data.

Allometric models:
    - NDVI-to-AGB exponential: AGB = a * exp(b * NDVI)
    - SAR backscatter-to-AGB power: AGB = a * sigma0^b
    - NDVI-to-canopy density linear: density_pct = slope * NDVI + intercept
    - AGB-to-height power: H = a * AGB^b

Data sources:
    - Avitabile et al. (2016) - Pan-tropical AGB map reconciliation
    - Saatchi et al. (2011) - Benchmark map of forest carbon stocks
    - Baccini et al. (2012) - Tropical forest carbon stocks from GLAS
    - Chave et al. (2014) - Improved allometric models for tropical trees
    - Mitchard et al. (2018) - SAR AGB estimation in tropical forests
    - Potapov et al. (2021) - Global canopy height mapping
    - Hansen et al. (2013) - Global forest cover change
    - Bouvet et al. (2018) - C-band SAR backscatter for forest AGB

EUDR Relevance:
    Biomass estimation supports EUDR compliance by:
    1. Quantifying carbon stock changes as degradation evidence
    2. Distinguishing primary forest from secondary regrowth
    3. Validating forest classification through structural parameters
    4. Providing quantitative evidence for Article 9 due diligence

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# NDVI-to-AGB allometric equations
# ---------------------------------------------------------------------------
#
# Exponential model: AGB (Mg/ha) = ndvi_to_agb_a * exp(ndvi_to_agb_b * NDVI)
#
# Each entry provides:
#   ndvi_to_agb_a         - Exponential coefficient a
#   ndvi_to_agb_b         - Exponential coefficient b
#   source                - Literature reference
#   r_squared             - Model fit quality (0.0-1.0)
#   valid_ndvi_range      - (min, max) NDVI range for reliable estimates
#   saturation_agb        - AGB (Mg/ha) above which estimate unreliable
#
# The saturation point reflects the well-known signal saturation of
# optical NDVI for dense tropical forests (typically >200 Mg/ha).

ALLOMETRIC_EQUATIONS: Dict[str, Dict[str, Any]] = {
    "tropical_moist_broadleaf": {
        "ndvi_to_agb_a": 2.5,
        "ndvi_to_agb_b": 6.2,
        "source": "Avitabile et al. (2016), Saatchi et al. (2011)",
        "r_squared": 0.68,
        "valid_ndvi_range": (0.30, 0.95),
        "saturation_agb": 350.0,
    },
    "tropical_dry_broadleaf": {
        "ndvi_to_agb_a": 3.0,
        "ndvi_to_agb_b": 5.5,
        "source": "Baccini et al. (2012), Chave et al. (2014)",
        "r_squared": 0.72,
        "valid_ndvi_range": (0.20, 0.90),
        "saturation_agb": 250.0,
    },
    "tropical_coniferous": {
        "ndvi_to_agb_a": 3.2,
        "ndvi_to_agb_b": 5.0,
        "source": "Chave et al. (2014)",
        "r_squared": 0.65,
        "valid_ndvi_range": (0.25, 0.85),
        "saturation_agb": 200.0,
    },
    "temperate_broadleaf_mixed": {
        "ndvi_to_agb_a": 2.8,
        "ndvi_to_agb_b": 5.8,
        "source": "Saatchi et al. (2011), Pan et al. (2011)",
        "r_squared": 0.70,
        "valid_ndvi_range": (0.25, 0.92),
        "saturation_agb": 300.0,
    },
    "temperate_coniferous": {
        "ndvi_to_agb_a": 2.6,
        "ndvi_to_agb_b": 5.5,
        "source": "Pan et al. (2011), Keith et al. (2009)",
        "r_squared": 0.66,
        "valid_ndvi_range": (0.20, 0.88),
        "saturation_agb": 350.0,
    },
    "boreal_taiga": {
        "ndvi_to_agb_a": 4.0,
        "ndvi_to_agb_b": 4.5,
        "source": "Thurner et al. (2014)",
        "r_squared": 0.60,
        "valid_ndvi_range": (0.15, 0.75),
        "saturation_agb": 150.0,
    },
    "tropical_grassland_savanna": {
        "ndvi_to_agb_a": 5.0,
        "ndvi_to_agb_b": 3.8,
        "source": "Baccini et al. (2012), Bouvet et al. (2018)",
        "r_squared": 0.58,
        "valid_ndvi_range": (0.15, 0.80),
        "saturation_agb": 80.0,
    },
    "temperate_grassland": {
        "ndvi_to_agb_a": 5.5,
        "ndvi_to_agb_b": 3.5,
        "source": "Scurlock and Hall (1998)",
        "r_squared": 0.55,
        "valid_ndvi_range": (0.15, 0.75),
        "saturation_agb": 40.0,
    },
    "montane_grassland": {
        "ndvi_to_agb_a": 4.5,
        "ndvi_to_agb_b": 4.0,
        "source": "Baccini et al. (2012)",
        "r_squared": 0.57,
        "valid_ndvi_range": (0.15, 0.78),
        "saturation_agb": 60.0,
    },
    "mediterranean_woodland": {
        "ndvi_to_agb_a": 3.5,
        "ndvi_to_agb_b": 4.8,
        "source": "Chirici et al. (2020)",
        "r_squared": 0.62,
        "valid_ndvi_range": (0.18, 0.82),
        "saturation_agb": 120.0,
    },
    "mangrove": {
        "ndvi_to_agb_a": 3.0,
        "ndvi_to_agb_b": 5.8,
        "source": "Simard et al. (2019), Fatoyinbo et al. (2018)",
        "r_squared": 0.64,
        "valid_ndvi_range": (0.20, 0.85),
        "saturation_agb": 300.0,
    },
    "flooded_grassland": {
        "ndvi_to_agb_a": 6.0,
        "ndvi_to_agb_b": 3.2,
        "source": "Baccini et al. (2012)",
        "r_squared": 0.50,
        "valid_ndvi_range": (0.10, 0.75),
        "saturation_agb": 50.0,
    },
    "desert_xeric": {
        "ndvi_to_agb_a": 8.0,
        "ndvi_to_agb_b": 2.8,
        "source": "Tucker et al. (2005)",
        "r_squared": 0.45,
        "valid_ndvi_range": (0.08, 0.55),
        "saturation_agb": 25.0,
    },
    "tropical_plantation": {
        "ndvi_to_agb_a": 3.5,
        "ndvi_to_agb_b": 5.2,
        "source": "Kho and Jepsen (2015), Li et al. (2017)",
        "r_squared": 0.70,
        "valid_ndvi_range": (0.25, 0.90),
        "saturation_agb": 150.0,
    },
    "agroforestry_system": {
        "ndvi_to_agb_a": 4.0,
        "ndvi_to_agb_b": 4.5,
        "source": "Nair et al. (2009), Albrecht and Kandji (2003)",
        "r_squared": 0.58,
        "valid_ndvi_range": (0.20, 0.85),
        "saturation_agb": 120.0,
    },
    "degraded_secondary": {
        "ndvi_to_agb_a": 4.0,
        "ndvi_to_agb_b": 4.8,
        "source": "Poorter et al. (2016)",
        "r_squared": 0.60,
        "valid_ndvi_range": (0.15, 0.82),
        "saturation_agb": 150.0,
    },
}


# ---------------------------------------------------------------------------
# SAR backscatter-to-AGB coefficients
# ---------------------------------------------------------------------------
#
# Power model: AGB (Mg/ha) = a * sigma0^b
#
# Separate coefficients for VV and VH polarizations (Sentinel-1 C-band).
# SAR is particularly useful where optical NDVI saturates (>200 Mg/ha)
# or where persistent cloud cover prevents optical observation.
#
# Each entry provides:
#   vv_a, vv_b           - AGB = vv_a * sigma0_VV^vv_b
#   vh_a, vh_b           - AGB = vh_a * sigma0_VH^vh_b
#   saturation_point     - AGB (Mg/ha) above which SAR estimate saturates
#   source               - Literature reference

SAR_COEFFICIENTS: Dict[str, Dict[str, Any]] = {
    "tropical_moist_broadleaf": {
        "vv_a": 45.0,
        "vv_b": 1.8,
        "vh_a": 38.0,
        "vh_b": 2.1,
        "saturation_point": 150.0,
        "source": "Mitchard et al. (2018), Bouvet et al. (2018)",
    },
    "tropical_dry_broadleaf": {
        "vv_a": 40.0,
        "vv_b": 1.7,
        "vh_a": 35.0,
        "vh_b": 2.0,
        "saturation_point": 120.0,
        "source": "Mitchard et al. (2018)",
    },
    "tropical_coniferous": {
        "vv_a": 38.0,
        "vv_b": 1.6,
        "vh_a": 32.0,
        "vh_b": 1.9,
        "saturation_point": 110.0,
        "source": "Bouvet et al. (2018)",
    },
    "temperate_broadleaf_mixed": {
        "vv_a": 42.0,
        "vv_b": 1.7,
        "vh_a": 36.0,
        "vh_b": 2.0,
        "saturation_point": 140.0,
        "source": "Santoro et al. (2021)",
    },
    "temperate_coniferous": {
        "vv_a": 40.0,
        "vv_b": 1.6,
        "vh_a": 34.0,
        "vh_b": 1.9,
        "saturation_point": 150.0,
        "source": "Santoro et al. (2021)",
    },
    "boreal_taiga": {
        "vv_a": 35.0,
        "vv_b": 1.5,
        "vh_a": 30.0,
        "vh_b": 1.8,
        "saturation_point": 100.0,
        "source": "Santoro et al. (2021), Thurner et al. (2014)",
    },
    "tropical_grassland_savanna": {
        "vv_a": 30.0,
        "vv_b": 1.4,
        "vh_a": 25.0,
        "vh_b": 1.7,
        "saturation_point": 60.0,
        "source": "Bouvet et al. (2018)",
    },
    "temperate_grassland": {
        "vv_a": 28.0,
        "vv_b": 1.3,
        "vh_a": 22.0,
        "vh_b": 1.5,
        "saturation_point": 40.0,
        "source": "Santoro et al. (2021)",
    },
    "montane_grassland": {
        "vv_a": 30.0,
        "vv_b": 1.4,
        "vh_a": 25.0,
        "vh_b": 1.6,
        "saturation_point": 50.0,
        "source": "Santoro et al. (2021)",
    },
    "mediterranean_woodland": {
        "vv_a": 35.0,
        "vv_b": 1.5,
        "vh_a": 30.0,
        "vh_b": 1.8,
        "saturation_point": 80.0,
        "source": "Chirici et al. (2020)",
    },
    "mangrove": {
        "vv_a": 40.0,
        "vv_b": 1.7,
        "vh_a": 35.0,
        "vh_b": 2.0,
        "saturation_point": 180.0,
        "source": "Simard et al. (2019)",
    },
    "flooded_grassland": {
        "vv_a": 25.0,
        "vv_b": 1.2,
        "vh_a": 20.0,
        "vh_b": 1.4,
        "saturation_point": 40.0,
        "source": "Bouvet et al. (2018)",
    },
    "desert_xeric": {
        "vv_a": 20.0,
        "vv_b": 1.1,
        "vh_a": 16.0,
        "vh_b": 1.3,
        "saturation_point": 20.0,
        "source": "Santoro et al. (2021)",
    },
    "tropical_plantation": {
        "vv_a": 38.0,
        "vv_b": 1.6,
        "vh_a": 32.0,
        "vh_b": 1.9,
        "saturation_point": 100.0,
        "source": "Li et al. (2017)",
    },
    "agroforestry_system": {
        "vv_a": 32.0,
        "vv_b": 1.5,
        "vh_a": 27.0,
        "vh_b": 1.7,
        "saturation_point": 80.0,
        "source": "Bouvet et al. (2018)",
    },
    "degraded_secondary": {
        "vv_a": 35.0,
        "vv_b": 1.5,
        "vh_a": 30.0,
        "vh_b": 1.8,
        "saturation_point": 100.0,
        "source": "Poorter et al. (2016)",
    },
}


# ---------------------------------------------------------------------------
# NDVI-to-canopy density regression coefficients
# ---------------------------------------------------------------------------
#
# Linear model: density_pct = slope * NDVI + intercept
#
# Provides a simple, deterministic canopy density estimate from NDVI.
# Biome-specific calibration is necessary because the same NDVI value
# implies different canopy densities across biome types.

NDVI_REGRESSION_COEFFICIENTS: Dict[str, Dict[str, Any]] = {
    "tropical_moist_broadleaf": {
        "slope": 140.0,
        "intercept": -20.0,
        "r_squared": 0.82,
        "source": "Hansen et al. (2013), Potapov et al. (2021)",
    },
    "tropical_dry_broadleaf": {
        "slope": 130.0,
        "intercept": -18.0,
        "r_squared": 0.78,
        "source": "Hansen et al. (2013)",
    },
    "tropical_coniferous": {
        "slope": 125.0,
        "intercept": -15.0,
        "r_squared": 0.75,
        "source": "Hansen et al. (2013)",
    },
    "temperate_broadleaf_mixed": {
        "slope": 135.0,
        "intercept": -17.0,
        "r_squared": 0.80,
        "source": "Potapov et al. (2021)",
    },
    "temperate_coniferous": {
        "slope": 128.0,
        "intercept": -14.0,
        "r_squared": 0.76,
        "source": "Potapov et al. (2021)",
    },
    "boreal_taiga": {
        "slope": 120.0,
        "intercept": -10.0,
        "r_squared": 0.72,
        "source": "Thurner et al. (2014)",
    },
    "tropical_grassland_savanna": {
        "slope": 115.0,
        "intercept": -12.0,
        "r_squared": 0.65,
        "source": "Hansen et al. (2013)",
    },
    "temperate_grassland": {
        "slope": 110.0,
        "intercept": -10.0,
        "r_squared": 0.60,
        "source": "Potapov et al. (2021)",
    },
    "montane_grassland": {
        "slope": 118.0,
        "intercept": -12.0,
        "r_squared": 0.63,
        "source": "Potapov et al. (2021)",
    },
    "mediterranean_woodland": {
        "slope": 118.0,
        "intercept": -12.0,
        "r_squared": 0.68,
        "source": "Chirici et al. (2020)",
    },
    "mangrove": {
        "slope": 122.0,
        "intercept": -10.0,
        "r_squared": 0.70,
        "source": "Simard et al. (2019)",
    },
    "flooded_grassland": {
        "slope": 110.0,
        "intercept": -8.0,
        "r_squared": 0.55,
        "source": "Hansen et al. (2013)",
    },
    "desert_xeric": {
        "slope": 100.0,
        "intercept": -5.0,
        "r_squared": 0.50,
        "source": "Tucker et al. (2005)",
    },
    "tropical_plantation": {
        "slope": 132.0,
        "intercept": -16.0,
        "r_squared": 0.78,
        "source": "Li et al. (2017)",
    },
    "agroforestry_system": {
        "slope": 125.0,
        "intercept": -14.0,
        "r_squared": 0.68,
        "source": "Nair et al. (2009)",
    },
    "degraded_secondary": {
        "slope": 118.0,
        "intercept": -12.0,
        "r_squared": 0.65,
        "source": "Poorter et al. (2016)",
    },
}


# ---------------------------------------------------------------------------
# AGB-to-height allometric equations
# ---------------------------------------------------------------------------
#
# Power model: H (m) = height_from_agb_a * AGB^height_from_agb_b
#
# Provides deterministic canopy height estimation from AGB estimates.
# These relationships are derived from GEDI L2A canopy height
# measurements cross-referenced with AGB maps.

HEIGHT_ALLOMETRIC: Dict[str, Dict[str, Any]] = {
    "tropical_moist_broadleaf": {
        "height_from_agb_a": 2.50,
        "height_from_agb_b": 0.42,
        "source": "Potapov et al. (2021), Chave et al. (2014)",
    },
    "tropical_dry_broadleaf": {
        "height_from_agb_a": 2.80,
        "height_from_agb_b": 0.38,
        "source": "Chave et al. (2014)",
    },
    "tropical_coniferous": {
        "height_from_agb_a": 3.00,
        "height_from_agb_b": 0.36,
        "source": "Chave et al. (2014)",
    },
    "temperate_broadleaf_mixed": {
        "height_from_agb_a": 2.60,
        "height_from_agb_b": 0.40,
        "source": "Potapov et al. (2021)",
    },
    "temperate_coniferous": {
        "height_from_agb_a": 2.40,
        "height_from_agb_b": 0.43,
        "source": "Potapov et al. (2021), Keith et al. (2009)",
    },
    "boreal_taiga": {
        "height_from_agb_a": 3.20,
        "height_from_agb_b": 0.35,
        "source": "Thurner et al. (2014)",
    },
    "tropical_grassland_savanna": {
        "height_from_agb_a": 2.00,
        "height_from_agb_b": 0.40,
        "source": "Baccini et al. (2012)",
    },
    "temperate_grassland": {
        "height_from_agb_a": 1.80,
        "height_from_agb_b": 0.38,
        "source": "Potapov et al. (2021)",
    },
    "montane_grassland": {
        "height_from_agb_a": 2.00,
        "height_from_agb_b": 0.38,
        "source": "Potapov et al. (2021)",
    },
    "mediterranean_woodland": {
        "height_from_agb_a": 2.50,
        "height_from_agb_b": 0.38,
        "source": "Chirici et al. (2020)",
    },
    "mangrove": {
        "height_from_agb_a": 2.20,
        "height_from_agb_b": 0.42,
        "source": "Simard et al. (2019)",
    },
    "flooded_grassland": {
        "height_from_agb_a": 1.60,
        "height_from_agb_b": 0.38,
        "source": "Baccini et al. (2012)",
    },
    "desert_xeric": {
        "height_from_agb_a": 1.50,
        "height_from_agb_b": 0.35,
        "source": "Tucker et al. (2005)",
    },
    "tropical_plantation": {
        "height_from_agb_a": 2.80,
        "height_from_agb_b": 0.38,
        "source": "Li et al. (2017)",
    },
    "agroforestry_system": {
        "height_from_agb_a": 2.40,
        "height_from_agb_b": 0.38,
        "source": "Nair et al. (2009)",
    },
    "degraded_secondary": {
        "height_from_agb_a": 2.60,
        "height_from_agb_b": 0.38,
        "source": "Poorter et al. (2016)",
    },
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_allometric_equation(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve the NDVI-to-AGB allometric equation for a specific biome.

    Args:
        biome: Biome name (e.g., 'tropical_moist_broadleaf').

    Returns:
        Dictionary with equation coefficients, source, r_squared,
        valid_ndvi_range, and saturation_agb. Returns None if the
        biome is not recognized.

    Example:
        >>> eq = get_allometric_equation("tropical_moist_broadleaf")
        >>> eq["ndvi_to_agb_a"]
        2.5
    """
    return ALLOMETRIC_EQUATIONS.get(biome)


def get_sar_coefficients(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve SAR backscatter-to-AGB coefficients for a specific biome.

    Args:
        biome: Biome name.

    Returns:
        Dictionary with VV and VH coefficients, saturation point,
        and source reference. Returns None if biome is not recognized.
    """
    return SAR_COEFFICIENTS.get(biome)


def get_ndvi_regression(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve NDVI-to-canopy density regression coefficients.

    Args:
        biome: Biome name.

    Returns:
        Dictionary with slope, intercept, r_squared, and source.
        Returns None if biome is not recognized.
    """
    return NDVI_REGRESSION_COEFFICIENTS.get(biome)


def get_height_allometric(biome: str) -> Optional[Dict[str, Any]]:
    """Retrieve AGB-to-height allometric equation for a specific biome.

    Args:
        biome: Biome name.

    Returns:
        Dictionary with height_from_agb_a, height_from_agb_b, and
        source. Returns None if biome is not recognized.
    """
    return HEIGHT_ALLOMETRIC.get(biome)


def estimate_agb_from_ndvi(
    ndvi: float,
    biome: str,
) -> Optional[float]:
    """Estimate above-ground biomass from an NDVI value.

    Uses the biome-specific exponential model:
        AGB = a * exp(b * NDVI)

    Returns None if the NDVI is outside the valid range for the biome
    or if the biome is not recognized.

    Args:
        ndvi: NDVI value.
        biome: Biome name.

    Returns:
        Estimated AGB in Mg/ha, capped at the saturation point,
        or None if inputs are invalid.
    """
    import math

    eq = ALLOMETRIC_EQUATIONS.get(biome)
    if eq is None:
        return None

    min_ndvi, max_ndvi = eq["valid_ndvi_range"]
    if ndvi < min_ndvi or ndvi > max_ndvi:
        return None

    agb = eq["ndvi_to_agb_a"] * math.exp(eq["ndvi_to_agb_b"] * ndvi)
    return min(agb, eq["saturation_agb"])


def estimate_height_from_agb(
    agb: float,
    biome: str,
) -> Optional[float]:
    """Estimate canopy height from above-ground biomass.

    Uses the biome-specific power model:
        H = a * AGB^b

    Args:
        agb: Above-ground biomass in Mg/ha (must be > 0).
        biome: Biome name.

    Returns:
        Estimated canopy height in metres, or None if biome is
        not recognized or AGB is non-positive.
    """
    if agb <= 0.0:
        return None

    ht = HEIGHT_ALLOMETRIC.get(biome)
    if ht is None:
        return None

    return ht["height_from_agb_a"] * (agb ** ht["height_from_agb_b"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ALLOMETRIC_EQUATIONS",
    "SAR_COEFFICIENTS",
    "NDVI_REGRESSION_COEFFICIENTS",
    "HEIGHT_ALLOMETRIC",
    "get_allometric_equation",
    "get_sar_coefficients",
    "get_ndvi_regression",
    "get_height_allometric",
    "estimate_agb_from_ndvi",
    "estimate_height_from_agb",
]
