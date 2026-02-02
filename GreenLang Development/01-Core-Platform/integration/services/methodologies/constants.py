# -*- coding: utf-8 -*-
"""
Methodologies Constants Module

This module provides scientific constants, lookup tables, and reference data
for uncertainty quantification and life cycle assessment calculations.

Key Features:
- ILCD Pedigree Matrix lookup tables (5 dimensions)
- GWP (Global Warming Potential) constants for AR5 and AR6
- Default uncertainty values by material/product category
- Quality scoring thresholds and ranges

References:
- ILCD Handbook (2010): https://eplca.jrc.ec.europa.eu/ilcd.html
- IPCC AR5 (2013): Fifth Assessment Report
- IPCC AR6 (2021): Sixth Assessment Report
- GHG Protocol: Corporate Value Chain (Scope 3) Accounting and Reporting Standard
- ISO 14044:2006: Environmental management - Life cycle assessment

Version: 1.0.0
Date: 2025-10-30
"""

from typing import Dict, List, Tuple
from enum import Enum


# ============================================================================
# ILCD PEDIGREE MATRIX - UNCERTAINTY FACTORS
# ============================================================================
# Reference: ILCD Handbook (2010), Chapter 5: Data Quality Guidelines
# Each dimension has 5 quality levels (1=excellent, 5=poor)
# Values represent the uncertainty factor (geometric standard deviation)

class PedigreeIndicator(str, Enum):
    """ILCD Pedigree Matrix indicators (5 dimensions)."""
    RELIABILITY = "reliability"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"


# Reliability: Verification and source quality
# 1: Verified data based on measurements
# 2: Verified data partly based on assumptions OR non-verified data based on measurements
# 3: Non-verified data partly based on qualified estimates
# 4: Qualified estimate (e.g., by industrial expert)
# 5: Non-qualified estimate
RELIABILITY_FACTORS: Dict[int, float] = {
    1: 1.00,  # Excellent: Verified data, measurements
    2: 1.05,  # Good: Verified data with some assumptions
    3: 1.10,  # Fair: Non-verified data with estimates
    4: 1.20,  # Poor: Qualified estimate
    5: 1.50,  # Very Poor: Non-qualified estimate
}

RELIABILITY_DESCRIPTIONS: Dict[int, str] = {
    1: "Verified data based on measurements",
    2: "Verified data partly based on assumptions OR non-verified data based on measurements",
    3: "Non-verified data partly based on qualified estimates",
    4: "Qualified estimate (e.g., by industrial expert)",
    5: "Non-qualified estimate",
}


# Completeness: Sample size and representativeness
# 1: Representative data from sufficient sample of sites over adequate period
# 2: Representative data from smaller number of sites but adequate period
# 3: Representative data from adequate number of sites but shorter period
# 4: Representative data from smaller number of sites and shorter period
# 5: Representativeness unknown or incomplete data
COMPLETENESS_FACTORS: Dict[int, float] = {
    1: 1.00,  # Excellent: Sufficient sample, adequate period
    2: 1.02,  # Good: Smaller sample, adequate period
    3: 1.05,  # Fair: Adequate sample, shorter period
    4: 1.10,  # Poor: Smaller sample, shorter period
    5: 1.20,  # Very Poor: Unknown representativeness
}

COMPLETENESS_DESCRIPTIONS: Dict[int, str] = {
    1: "Representative data from sufficient sample of sites over adequate period",
    2: "Representative data from smaller number of sites but adequate period",
    3: "Representative data from adequate number of sites but shorter period",
    4: "Representative data from smaller number of sites and shorter period",
    5: "Representativeness unknown or incomplete data",
}


# Temporal Correlation: Age of data
# 1: Less than 3 years of difference to reference year
# 2: Less than 6 years of difference
# 3: Less than 10 years of difference
# 4: Less than 15 years of difference
# 5: Age unknown or more than 15 years of difference
TEMPORAL_FACTORS: Dict[int, float] = {
    1: 1.00,  # Excellent: <3 years old
    2: 1.03,  # Good: 3-6 years old
    3: 1.10,  # Fair: 6-10 years old
    4: 1.20,  # Poor: 10-15 years old
    5: 1.50,  # Very Poor: >15 years or unknown
}

TEMPORAL_DESCRIPTIONS: Dict[int, str] = {
    1: "Less than 3 years of difference to reference year",
    2: "Less than 6 years of difference to reference year",
    3: "Less than 10 years of difference to reference year",
    4: "Less than 15 years of difference to reference year",
    5: "Age unknown or more than 15 years of difference",
}


# Geographical Correlation: Geographical representativeness
# 1: Data from area under study
# 2: Average data from larger area in which area under study is included
# 3: Data from area with similar production conditions
# 4: Data from area with slightly similar production conditions
# 5: Data from unknown or distinctly different area
GEOGRAPHICAL_FACTORS: Dict[int, float] = {
    1: 1.00,  # Excellent: Same area
    2: 1.01,  # Good: Larger area including study area
    3: 1.02,  # Fair: Similar production conditions
    4: 1.10,  # Poor: Slightly similar conditions
    5: 1.20,  # Very Poor: Unknown or different area
}

GEOGRAPHICAL_DESCRIPTIONS: Dict[int, str] = {
    1: "Data from area under study",
    2: "Average data from larger area in which area under study is included",
    3: "Data from area with similar production conditions",
    4: "Data from area with slightly similar production conditions",
    5: "Data from unknown or distinctly different area",
}


# Technological Correlation: Technology representativeness
# 1: Data from enterprises, processes, materials under study
# 2: Data from processes and materials under study but different enterprises
# 3: Data from processes and materials under study but different technology
# 4: Data on related processes or materials but same technology
# 5: Data on related processes or materials but different technology
TECHNOLOGICAL_FACTORS: Dict[int, float] = {
    1: 1.00,  # Excellent: Same technology
    2: 1.05,  # Good: Same process, different enterprise
    3: 1.20,  # Fair: Same process, different technology
    4: 1.50,  # Poor: Related process, same technology
    5: 2.00,  # Very Poor: Related process, different technology
}

TECHNOLOGICAL_DESCRIPTIONS: Dict[int, str] = {
    1: "Data from enterprises, processes, and materials under study",
    2: "Data from processes and materials under study but from different enterprises",
    3: "Data from processes and materials under study but from different technology",
    4: "Data on related processes or materials but same technology",
    5: "Data on related processes or materials but different technology",
}


# Combined Pedigree Matrix
PEDIGREE_MATRIX: Dict[PedigreeIndicator, Dict[int, float]] = {
    PedigreeIndicator.RELIABILITY: RELIABILITY_FACTORS,
    PedigreeIndicator.COMPLETENESS: COMPLETENESS_FACTORS,
    PedigreeIndicator.TEMPORAL: TEMPORAL_FACTORS,
    PedigreeIndicator.GEOGRAPHICAL: GEOGRAPHICAL_FACTORS,
    PedigreeIndicator.TECHNOLOGICAL: TECHNOLOGICAL_FACTORS,
}

PEDIGREE_DESCRIPTIONS: Dict[PedigreeIndicator, Dict[int, str]] = {
    PedigreeIndicator.RELIABILITY: RELIABILITY_DESCRIPTIONS,
    PedigreeIndicator.COMPLETENESS: COMPLETENESS_DESCRIPTIONS,
    PedigreeIndicator.TEMPORAL: TEMPORAL_DESCRIPTIONS,
    PedigreeIndicator.GEOGRAPHICAL: GEOGRAPHICAL_DESCRIPTIONS,
    PedigreeIndicator.TECHNOLOGICAL: TECHNOLOGICAL_DESCRIPTIONS,
}


# ============================================================================
# GLOBAL WARMING POTENTIAL (GWP) CONSTANTS
# ============================================================================
# Reference: IPCC Assessment Reports
# Time horizon: 100 years (GWP100)

class GWPVersion(str, Enum):
    """IPCC Assessment Report versions."""
    AR5 = "AR5"  # Fifth Assessment Report (2013)
    AR6 = "AR6"  # Sixth Assessment Report (2021)


# IPCC AR5 (2013) - Including climate-carbon feedbacks
GWP_AR5: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 28.0,  # Fossil CH4
    "CH4_biogenic": 30.0,  # Biogenic CH4
    "N2O": 265.0,
    "HFC-134a": 1300.0,
    "HFC-32": 677.0,
    "HFC-125": 3170.0,
    "HFC-143a": 4800.0,
    "HFC-152a": 138.0,
    "HFC-227ea": 3350.0,
    "HFC-23": 12400.0,
    "HFC-245fa": 858.0,
    "SF6": 23500.0,
    "NF3": 16100.0,
    "CF4": 6630.0,
    "C2F6": 11100.0,
}

# IPCC AR6 (2021) - Including climate-carbon feedbacks
GWP_AR6: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 29.8,  # Fossil CH4
    "CH4_biogenic": 27.2,  # Biogenic CH4
    "N2O": 273.0,
    "HFC-134a": 1530.0,
    "HFC-32": 771.0,
    "HFC-125": 3740.0,
    "HFC-143a": 5810.0,
    "HFC-152a": 164.0,
    "HFC-227ea": 3600.0,
    "HFC-23": 14600.0,
    "HFC-245fa": 962.0,
    "SF6": 25200.0,
    "NF3": 17400.0,
    "CF4": 7380.0,
    "C2F6": 12400.0,
}

# Default GWP version (use AR5 for compatibility with most databases)
DEFAULT_GWP_VERSION = GWPVersion.AR5


# ============================================================================
# DEFAULT UNCERTAINTY VALUES BY CATEGORY
# ============================================================================
# Reference: ecoinvent methodology, GHG Protocol guidance
# Values represent relative standard deviation (coefficient of variation)

# Category-based default uncertainties (as percentage of mean)
DEFAULT_UNCERTAINTIES: Dict[str, float] = {
    # Raw materials and commodities
    "metals": 0.15,  # 15% uncertainty
    "plastics": 0.20,  # 20% uncertainty
    "chemicals": 0.25,  # 25% uncertainty
    "textiles": 0.30,  # 30% uncertainty
    "wood_products": 0.20,
    "minerals": 0.15,
    "glass": 0.15,
    "ceramics": 0.20,
    "paper_cardboard": 0.20,

    # Energy and utilities
    "electricity": 0.10,  # 10% uncertainty (well-documented)
    "natural_gas": 0.08,
    "coal": 0.12,
    "fuel_oil": 0.10,
    "diesel": 0.08,
    "gasoline": 0.08,
    "renewable_energy": 0.15,

    # Transportation
    "road_transport": 0.15,
    "rail_transport": 0.20,
    "sea_freight": 0.25,
    "air_freight": 0.20,
    "pipeline_transport": 0.15,

    # Services
    "business_services": 0.40,  # High uncertainty
    "financial_services": 0.50,
    "it_services": 0.35,
    "professional_services": 0.40,

    # Waste treatment
    "landfill": 0.25,
    "incineration": 0.20,
    "recycling": 0.30,
    "composting": 0.30,

    # Agriculture and food
    "crops": 0.35,
    "livestock": 0.40,
    "food_processing": 0.25,
    "beverages": 0.20,

    # Construction and infrastructure
    "concrete": 0.15,
    "steel_construction": 0.15,
    "building_materials": 0.20,
    "construction_services": 0.30,

    # Default (unknown category)
    "default": 0.50,  # 50% uncertainty for unknown categories
}


# Tier-based uncertainty multipliers
TIER_UNCERTAINTY_MULTIPLIERS: Dict[int, float] = {
    1: 1.0,   # Tier 1: Primary data (no increase)
    2: 1.5,   # Tier 2: Secondary data (50% increase)
    3: 2.0,   # Tier 3: Estimated data (100% increase)
}


# ============================================================================
# DATA QUALITY INDEX (DQI) THRESHOLDS
# ============================================================================
# DQI scoring system (0-100 scale)

DQI_QUALITY_LABELS: Dict[str, Tuple[float, float]] = {
    "Excellent": (90.0, 100.0),  # DQI >= 90
    "Good": (70.0, 89.9),        # 70 <= DQI < 90
    "Fair": (50.0, 69.9),        # 50 <= DQI < 70
    "Poor": (0.0, 49.9),         # DQI < 50
}


# Factor source quality scores (0-100)
FACTOR_SOURCE_SCORES: Dict[str, float] = {
    # Primary sources (highest quality)
    "primary_measured": 100.0,
    "primary_calculated": 95.0,

    # Secondary sources - Tier 1
    "ecoinvent": 90.0,
    "gabi": 90.0,
    "idemat": 85.0,
    "agribalyse": 85.0,

    # Secondary sources - Tier 2
    "defra": 80.0,
    "epa": 80.0,
    "ghg_protocol": 80.0,
    "ipcc": 85.0,

    # Secondary sources - Tier 3
    "industry_average": 70.0,
    "literature": 75.0,
    "supplier_specific": 80.0,

    # Estimated values
    "expert_estimate": 60.0,
    "proxy": 50.0,
    "extrapolation": 55.0,

    # Unknown
    "unknown": 30.0,
}


# Pedigree score to DQI conversion
# Maps average pedigree score (1-5) to DQI score (0-100)
PEDIGREE_TO_DQI_MAPPING: Dict[float, float] = {
    1.0: 100.0,  # Perfect score
    1.5: 90.0,
    2.0: 75.0,
    2.5: 60.0,
    3.0: 50.0,
    3.5: 40.0,
    4.0: 30.0,
    4.5: 20.0,
    5.0: 10.0,   # Worst score
}


# ============================================================================
# MONTE CARLO SIMULATION PARAMETERS
# ============================================================================

# Default simulation parameters
MC_DEFAULT_ITERATIONS: int = 10_000
MC_MIN_ITERATIONS: int = 1_000
MC_MAX_ITERATIONS: int = 1_000_000

# Confidence intervals (percentiles)
MC_CONFIDENCE_LEVELS: List[float] = [0.05, 0.50, 0.95]  # p5, p50, p95

# Distribution types
class DistributionType(str, Enum):
    """Probability distribution types for Monte Carlo simulation."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# Default distribution by category
DEFAULT_DISTRIBUTIONS: Dict[str, DistributionType] = {
    "emissions": DistributionType.LOGNORMAL,  # Most environmental data
    "energy": DistributionType.LOGNORMAL,
    "waste": DistributionType.LOGNORMAL,
    "transport": DistributionType.NORMAL,
    "services": DistributionType.LOGNORMAL,
    "default": DistributionType.LOGNORMAL,
}


# ============================================================================
# UNCERTAINTY PROPAGATION RULES
# ============================================================================

# Minimum uncertainty (floor) - 1%
MIN_UNCERTAINTY: float = 0.01

# Maximum uncertainty (ceiling) - 500%
MAX_UNCERTAINTY: float = 5.00

# Basic uncertainty (when no better data available)
BASIC_UNCERTAINTY: float = 0.50  # 50%


# ============================================================================
# SENSITIVITY ANALYSIS THRESHOLDS
# ============================================================================

# Contribution threshold for sensitivity analysis (percentage)
SENSITIVITY_THRESHOLD: float = 0.01  # 1% contribution

# Top contributors to report
TOP_CONTRIBUTORS_COUNT: int = 10


# ============================================================================
# VALIDATION RANGES
# ============================================================================

# Valid ranges for pedigree scores
PEDIGREE_SCORE_MIN: int = 1
PEDIGREE_SCORE_MAX: int = 5

# Valid ranges for DQI scores
DQI_SCORE_MIN: float = 0.0
DQI_SCORE_MAX: float = 100.0

# Valid ranges for uncertainty (as decimal)
UNCERTAINTY_MIN: float = 0.0
UNCERTAINTY_MAX: float = 10.0  # 1000% (extreme cases)


# ============================================================================
# MODULE METADATA
# ============================================================================

__all__ = [
    # Enums
    "PedigreeIndicator",
    "GWPVersion",
    "DistributionType",

    # Pedigree Matrix
    "PEDIGREE_MATRIX",
    "PEDIGREE_DESCRIPTIONS",
    "RELIABILITY_FACTORS",
    "COMPLETENESS_FACTORS",
    "TEMPORAL_FACTORS",
    "GEOGRAPHICAL_FACTORS",
    "TECHNOLOGICAL_FACTORS",

    # GWP Constants
    "GWP_AR5",
    "GWP_AR6",
    "DEFAULT_GWP_VERSION",

    # Uncertainty Defaults
    "DEFAULT_UNCERTAINTIES",
    "TIER_UNCERTAINTY_MULTIPLIERS",
    "MIN_UNCERTAINTY",
    "MAX_UNCERTAINTY",
    "BASIC_UNCERTAINTY",

    # DQI Constants
    "DQI_QUALITY_LABELS",
    "FACTOR_SOURCE_SCORES",
    "PEDIGREE_TO_DQI_MAPPING",
    "DQI_SCORE_MIN",
    "DQI_SCORE_MAX",

    # Monte Carlo
    "MC_DEFAULT_ITERATIONS",
    "MC_MIN_ITERATIONS",
    "MC_MAX_ITERATIONS",
    "MC_CONFIDENCE_LEVELS",
    "DEFAULT_DISTRIBUTIONS",

    # Sensitivity Analysis
    "SENSITIVITY_THRESHOLD",
    "TOP_CONTRIBUTORS_COUNT",

    # Validation
    "PEDIGREE_SCORE_MIN",
    "PEDIGREE_SCORE_MAX",
    "UNCERTAINTY_MIN",
    "UNCERTAINTY_MAX",
]
