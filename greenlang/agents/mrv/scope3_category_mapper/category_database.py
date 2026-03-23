# -*- coding: utf-8 -*-
"""
CategoryDatabaseEngine - Classification code mapping tables and lookup engine.

This module implements the CategoryDatabaseEngine for AGENT-MRV-029
(Scope 3 Category Mapper, GL-MRV-X-040). It provides thread-safe singleton
access to comprehensive deterministic lookup tables that map organisational
classification codes to GHG Protocol Scope 3 categories (1-15).

Mapping Tables:
- NAICS 2022: All 20 two-digit sectors (11-92) with ~50 three-digit subsector
  overrides, each producing primary + secondary Scope 3 category assignments
- ISIC Rev 4: All 21 sections (A-U) with ~40 two-digit division overrides
- NAICS-ISIC concordance: Bidirectional cross-reference mapping
- GL Account Ranges: 15+ standard account ranges to categories
- Spend Keyword Dictionary: 500+ keywords with confidence-weighted category
  assignments across all 15 Scope 3 categories
- Category Info: Comprehensive metadata for each of the 15 categories

Zero-Hallucination Guarantee:
    All mappings are hardcoded deterministic lookup tables. NO LLM, ML, or
    probabilistic models are used. Every lookup returns data from frozen
    constant dictionaries sourced from Census Bureau (NAICS), UN (ISIC),
    and GHG Protocol Scope 3 Standard (category definitions).

Thread Safety:
    Uses the __new__ singleton pattern with threading.Lock to ensure only
    one instance is created across all threads.

Example:
    >>> engine = CategoryDatabaseEngine()
    >>> result = engine.lookup_naics("481")
    >>> result.primary_category
    <Scope3Category.CAT_6: 6>
    >>> result.confidence
    Decimal('0.92')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# =============================================================================
# AGENT METADATA
# =============================================================================

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_scm_"
MAPPING_VERSION: str = "2026.1.0"

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class Scope3Category(int, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_1 = 1   # Purchased Goods & Services
    CAT_2 = 2   # Capital Goods
    CAT_3 = 3   # Fuel & Energy Activities
    CAT_4 = 4   # Upstream Transportation & Distribution
    CAT_5 = 5   # Waste Generated in Operations
    CAT_6 = 6   # Business Travel
    CAT_7 = 7   # Employee Commuting
    CAT_8 = 8   # Upstream Leased Assets
    CAT_9 = 9   # Downstream Transportation & Distribution
    CAT_10 = 10  # Processing of Sold Products
    CAT_11 = 11  # Use of Sold Products
    CAT_12 = 12  # End-of-Life Treatment of Sold Products
    CAT_13 = 13  # Downstream Leased Assets
    CAT_14 = 14  # Franchises
    CAT_15 = 15  # Investments


class ValueChainDirection(str, Enum):
    """Value chain direction for a Scope 3 category."""

    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


# =============================================================================
# RESULT MODELS
# =============================================================================


class NAICSLookupResult(BaseModel):
    """Result from a NAICS code lookup."""

    naics_code: str = Field(..., description="Input NAICS code (2-6 digit)")
    matched_code: str = Field(..., description="Code that matched in the mapping table")
    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Secondary Scope 3 categories that may also apply"
    )
    confidence: Decimal = Field(..., description="Mapping confidence (0.0-1.0)")
    description: str = Field(..., description="Human-readable description of the mapping")
    provenance_hash: str = Field(..., description="SHA-256 hash of the lookup result")

    model_config = ConfigDict(frozen=True)


class ISICLookupResult(BaseModel):
    """Result from an ISIC Rev 4 code lookup."""

    isic_code: str = Field(..., description="Input ISIC code")
    matched_code: str = Field(..., description="Code that matched in the mapping table")
    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Secondary Scope 3 categories"
    )
    confidence: Decimal = Field(..., description="Mapping confidence (0.0-1.0)")
    description: str = Field(..., description="Human-readable description")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = ConfigDict(frozen=True)


class GLLookupResult(BaseModel):
    """Result from a GL account code lookup."""

    account_code: str = Field(..., description="Input GL account code")
    matched_range: str = Field(..., description="GL range that matched (e.g. '5000-5199')")
    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    secondary_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Secondary Scope 3 categories"
    )
    confidence: Decimal = Field(..., description="Mapping confidence")
    description: str = Field(..., description="Account type description")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = ConfigDict(frozen=True)


class KeywordLookupResult(BaseModel):
    """Result from a keyword lookup."""

    input_text: str = Field(..., description="Original input text")
    matched_keyword: str = Field(..., description="Best matching keyword")
    primary_category: Scope3Category = Field(..., description="Primary Scope 3 category")
    confidence: Decimal = Field(..., description="Keyword match confidence (0.0-1.0)")
    keyword_group: str = Field(..., description="Keyword group that matched")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = ConfigDict(frozen=True)


class CategoryInfo(BaseModel):
    """Comprehensive info for a single GHG Protocol Scope 3 category."""

    number: int = Field(..., ge=1, le=15, description="Category number (1-15)")
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Full category description")
    direction: ValueChainDirection = Field(..., description="Upstream or downstream")
    ghg_protocol_chapter: str = Field(..., description="GHG Protocol Standard chapter")
    reporter_role: str = Field(..., description="Role of the reporting company")
    typical_data_sources: List[str] = Field(
        default_factory=list,
        description="Typical data sources for this category"
    )
    downstream_agent: str = Field(
        ..., description="Agent ID that handles this category (MRV-014 to MRV-028)"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# TABLE 1: NAICS 2022 -> SCOPE 3 CATEGORY MAPPINGS
# =============================================================================

# Structure: naics_code -> (primary_category, secondary_categories, confidence, description)
# Two-digit sector codes provide broad mapping; three-digit subsector codes
# provide specialised overrides for transport, waste, finance, etc.

_NAICS_SECTOR_MAPPINGS: Dict[str, Dict[str, Any]] = {
    # ---- 2-digit sector-level mappings (all 20 NAICS sectors) ----
    "11": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.80"),
        "description": "Agriculture, Forestry, Fishing and Hunting",
    },
    "21": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_3],
        "confidence": Decimal("0.80"),
        "description": "Mining, Quarrying, and Oil and Gas Extraction",
    },
    "22": {
        "primary": Scope3Category.CAT_3,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Utilities",
    },
    "23": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.80"),
        "description": "Construction",
    },
    "31": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2, Scope3Category.CAT_10],
        "confidence": Decimal("0.78"),
        "description": "Manufacturing (Food, Beverage, Textile, Apparel)",
    },
    "32": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2, Scope3Category.CAT_10],
        "confidence": Decimal("0.78"),
        "description": "Manufacturing (Wood, Paper, Petroleum, Chemical, Plastics)",
    },
    "33": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2, Scope3Category.CAT_10],
        "confidence": Decimal("0.78"),
        "description": "Manufacturing (Metal, Machinery, Electronics, Transport Equip)",
    },
    "42": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_4],
        "confidence": Decimal("0.80"),
        "description": "Wholesale Trade",
    },
    "44": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.78"),
        "description": "Retail Trade (Motor Vehicle, Furniture, Electronics, Building)",
    },
    "45": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.78"),
        "description": "Retail Trade (Food, Health, Clothing, General, E-commerce)",
    },
    "48": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_6, Scope3Category.CAT_9],
        "confidence": Decimal("0.82"),
        "description": "Transportation (Air, Rail, Water, Truck, Transit, Pipeline)",
    },
    "49": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.82"),
        "description": "Postal Service, Couriers, Warehousing",
    },
    "51": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.78"),
        "description": "Information (Publishing, Broadcasting, Telecom, Data Processing)",
    },
    "52": {
        "primary": Scope3Category.CAT_15,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.85"),
        "description": "Finance and Insurance",
    },
    "53": {
        "primary": Scope3Category.CAT_8,
        "secondary": [Scope3Category.CAT_13, Scope3Category.CAT_2],
        "confidence": Decimal("0.80"),
        "description": "Real Estate and Rental and Leasing",
    },
    "54": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Professional, Scientific, and Technical Services",
    },
    "55": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_15],
        "confidence": Decimal("0.75"),
        "description": "Management of Companies and Enterprises",
    },
    "56": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_5],
        "confidence": Decimal("0.78"),
        "description": "Administrative and Support and Waste Management Services",
    },
    "61": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.78"),
        "description": "Educational Services",
    },
    "62": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.78"),
        "description": "Health Care and Social Assistance",
    },
    "71": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_6],
        "confidence": Decimal("0.75"),
        "description": "Arts, Entertainment, and Recreation",
    },
    "72": {
        "primary": Scope3Category.CAT_6,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.82"),
        "description": "Accommodation and Food Services",
    },
    "81": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.78"),
        "description": "Other Services (except Public Administration)",
    },
    "92": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.72"),
        "description": "Public Administration",
    },
}

# ---- 3-digit subsector overrides (~50 specialised mappings) ----
_NAICS_SUBSECTOR_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Agriculture
    "111": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Crop Production",
    },
    "112": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Animal Production and Aquaculture",
    },
    "113": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Forestry and Logging",
    },
    "114": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Fishing, Hunting and Trapping",
    },
    "115": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Support Activities for Agriculture and Forestry",
    },
    # Mining
    "211": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.88"),
        "description": "Oil and Gas Extraction",
    },
    "212": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_3],
        "confidence": Decimal("0.85"),
        "description": "Mining (except Oil and Gas)",
    },
    "213": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Support Activities for Mining",
    },
    # Utilities
    "221": {
        "primary": Scope3Category.CAT_3,
        "secondary": [],
        "confidence": Decimal("0.90"),
        "description": "Utilities (Electric Power, Natural Gas, Water, Sewage)",
    },
    # Construction
    "236": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.85"),
        "description": "Construction of Buildings",
    },
    "237": {
        "primary": Scope3Category.CAT_2,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Heavy and Civil Engineering Construction",
    },
    "238": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2],
        "confidence": Decimal("0.82"),
        "description": "Specialty Trade Contractors",
    },
    # Manufacturing - machinery and equipment subsectors
    "333": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.88"),
        "description": "Machinery Manufacturing",
    },
    "334": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.88"),
        "description": "Computer and Electronic Product Manufacturing",
    },
    "335": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.85"),
        "description": "Electrical Equipment, Appliance, and Component Manufacturing",
    },
    "336": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.88"),
        "description": "Transportation Equipment Manufacturing",
    },
    "337": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.82"),
        "description": "Furniture and Related Product Manufacturing",
    },
    "339": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2],
        "confidence": Decimal("0.80"),
        "description": "Miscellaneous Manufacturing",
    },
    # Manufacturing - process industries
    "311": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Food Manufacturing",
    },
    "312": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Beverage and Tobacco Product Manufacturing",
    },
    "313": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Textile Mills",
    },
    "314": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Textile Product Mills",
    },
    "315": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Apparel Manufacturing",
    },
    "316": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Leather and Allied Product Manufacturing",
    },
    "321": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Wood Product Manufacturing",
    },
    "322": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Paper Manufacturing",
    },
    "323": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Printing and Related Support Activities",
    },
    "324": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.88"),
        "description": "Petroleum and Coal Products Manufacturing",
    },
    "325": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10, Scope3Category.CAT_11],
        "confidence": Decimal("0.85"),
        "description": "Chemical Manufacturing",
    },
    "326": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.85"),
        "description": "Plastics and Rubber Products Manufacturing",
    },
    "327": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Nonmetallic Mineral Product Manufacturing",
    },
    "331": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Primary Metal Manufacturing",
    },
    "332": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2],
        "confidence": Decimal("0.82"),
        "description": "Fabricated Metal Product Manufacturing",
    },
    # Transportation subsectors (critical routing)
    "481": {
        "primary": Scope3Category.CAT_6,
        "secondary": [Scope3Category.CAT_4],
        "confidence": Decimal("0.92"),
        "description": "Air Transportation",
    },
    "482": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_6, Scope3Category.CAT_7],
        "confidence": Decimal("0.88"),
        "description": "Rail Transportation",
    },
    "483": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.88"),
        "description": "Water Transportation",
    },
    "484": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.92"),
        "description": "Truck Transportation",
    },
    "485": {
        "primary": Scope3Category.CAT_7,
        "secondary": [Scope3Category.CAT_6],
        "confidence": Decimal("0.85"),
        "description": "Transit and Ground Passenger Transportation",
    },
    "486": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_3],
        "confidence": Decimal("0.88"),
        "description": "Pipeline Transportation",
    },
    "487": {
        "primary": Scope3Category.CAT_6,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Scenic and Sightseeing Transportation",
    },
    "488": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.85"),
        "description": "Support Activities for Transportation",
    },
    "491": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.85"),
        "description": "Postal Service",
    },
    "492": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.88"),
        "description": "Couriers and Messengers",
    },
    "493": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.85"),
        "description": "Warehousing and Storage",
    },
    # Finance and Insurance subsectors
    "521": {
        "primary": Scope3Category.CAT_15,
        "secondary": [],
        "confidence": Decimal("0.90"),
        "description": "Monetary Authorities - Central Bank",
    },
    "522": {
        "primary": Scope3Category.CAT_15,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.88"),
        "description": "Credit Intermediation and Related Activities",
    },
    "523": {
        "primary": Scope3Category.CAT_15,
        "secondary": [],
        "confidence": Decimal("0.92"),
        "description": "Securities, Commodity Contracts, Other Financial Investments",
    },
    "524": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_15],
        "confidence": Decimal("0.82"),
        "description": "Insurance Carriers and Related Activities",
    },
    "525": {
        "primary": Scope3Category.CAT_15,
        "secondary": [],
        "confidence": Decimal("0.92"),
        "description": "Funds, Trusts, and Other Financial Vehicles",
    },
    # Real Estate
    "531": {
        "primary": Scope3Category.CAT_13,
        "secondary": [Scope3Category.CAT_8],
        "confidence": Decimal("0.88"),
        "description": "Real Estate",
    },
    "532": {
        "primary": Scope3Category.CAT_8,
        "secondary": [Scope3Category.CAT_13],
        "confidence": Decimal("0.88"),
        "description": "Rental and Leasing Services",
    },
    "533": {
        "primary": Scope3Category.CAT_8,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Lessors of Nonfinancial Intangible Assets",
    },
    # Waste Management (critical subsector)
    "562": {
        "primary": Scope3Category.CAT_5,
        "secondary": [Scope3Category.CAT_12],
        "confidence": Decimal("0.92"),
        "description": "Waste Management and Remediation Services",
    },
    # Accommodation and Food
    "721": {
        "primary": Scope3Category.CAT_6,
        "secondary": [],
        "confidence": Decimal("0.90"),
        "description": "Accommodation",
    },
    "722": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_14],
        "confidence": Decimal("0.80"),
        "description": "Food Services and Drinking Places",
    },
}


# =============================================================================
# TABLE 2: ISIC REV 4 -> SCOPE 3 CATEGORY MAPPINGS
# =============================================================================

# Section-level mappings (21 sections A-U)
_ISIC_SECTION_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "A": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.80"),
        "description": "Agriculture, forestry and fishing",
    },
    "B": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_3],
        "confidence": Decimal("0.80"),
        "description": "Mining and quarrying",
    },
    "C": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2, Scope3Category.CAT_10],
        "confidence": Decimal("0.78"),
        "description": "Manufacturing",
    },
    "D": {
        "primary": Scope3Category.CAT_3,
        "secondary": [],
        "confidence": Decimal("0.88"),
        "description": "Electricity, gas, steam and air conditioning supply",
    },
    "E": {
        "primary": Scope3Category.CAT_5,
        "secondary": [Scope3Category.CAT_3],
        "confidence": Decimal("0.85"),
        "description": "Water supply; sewerage, waste management and remediation",
    },
    "F": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.82"),
        "description": "Construction",
    },
    "G": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_4, Scope3Category.CAT_9],
        "confidence": Decimal("0.78"),
        "description": "Wholesale and retail trade; repair of motor vehicles",
    },
    "H": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_6, Scope3Category.CAT_9],
        "confidence": Decimal("0.82"),
        "description": "Transportation and storage",
    },
    "I": {
        "primary": Scope3Category.CAT_6,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.82"),
        "description": "Accommodation and food service activities",
    },
    "J": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.78"),
        "description": "Information and communication",
    },
    "K": {
        "primary": Scope3Category.CAT_15,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.85"),
        "description": "Financial and insurance activities",
    },
    "L": {
        "primary": Scope3Category.CAT_8,
        "secondary": [Scope3Category.CAT_13],
        "confidence": Decimal("0.82"),
        "description": "Real estate activities",
    },
    "M": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.80"),
        "description": "Professional, scientific and technical activities",
    },
    "N": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_5],
        "confidence": Decimal("0.78"),
        "description": "Administrative and support service activities",
    },
    "O": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.72"),
        "description": "Public administration and defence; compulsory social security",
    },
    "P": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.75"),
        "description": "Education",
    },
    "Q": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.75"),
        "description": "Human health and social work activities",
    },
    "R": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_6],
        "confidence": Decimal("0.75"),
        "description": "Arts, entertainment and recreation",
    },
    "S": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.72"),
        "description": "Other service activities",
    },
    "T": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_7],
        "confidence": Decimal("0.70"),
        "description": "Activities of households as employers",
    },
    "U": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.68"),
        "description": "Activities of extraterritorial organizations and bodies",
    },
}

# Division-level overrides (~40 two-digit ISIC divisions)
_ISIC_DIVISION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Agriculture
    "01": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Crop and animal production, hunting and related service activities",
    },
    "02": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Forestry and logging",
    },
    "03": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Fishing and aquaculture",
    },
    # Mining
    "05": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.88"),
        "description": "Mining of coal and lignite",
    },
    "06": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.90"),
        "description": "Extraction of crude petroleum and natural gas",
    },
    "07": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Mining of metal ores",
    },
    "08": {
        "primary": Scope3Category.CAT_1,
        "secondary": [],
        "confidence": Decimal("0.82"),
        "description": "Other mining and quarrying",
    },
    # Manufacturing overrides
    "10": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Manufacture of food products",
    },
    "11": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10],
        "confidence": Decimal("0.88"),
        "description": "Manufacture of beverages",
    },
    "19": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.90"),
        "description": "Manufacture of coke and refined petroleum products",
    },
    "20": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_10, Scope3Category.CAT_11],
        "confidence": Decimal("0.85"),
        "description": "Manufacture of chemicals and chemical products",
    },
    "26": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.88"),
        "description": "Manufacture of computer, electronic and optical products",
    },
    "27": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.85"),
        "description": "Manufacture of electrical equipment",
    },
    "28": {
        "primary": Scope3Category.CAT_2,
        "secondary": [],
        "confidence": Decimal("0.88"),
        "description": "Manufacture of machinery and equipment n.e.c.",
    },
    "29": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.90"),
        "description": "Manufacture of motor vehicles, trailers and semi-trailers",
    },
    "30": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_11],
        "confidence": Decimal("0.88"),
        "description": "Manufacture of other transport equipment",
    },
    # Utilities
    "35": {
        "primary": Scope3Category.CAT_3,
        "secondary": [],
        "confidence": Decimal("0.92"),
        "description": "Electricity, gas, steam and air conditioning supply",
    },
    "36": {
        "primary": Scope3Category.CAT_3,
        "secondary": [Scope3Category.CAT_5],
        "confidence": Decimal("0.85"),
        "description": "Water collection, treatment and supply",
    },
    "37": {
        "primary": Scope3Category.CAT_5,
        "secondary": [],
        "confidence": Decimal("0.90"),
        "description": "Sewerage",
    },
    "38": {
        "primary": Scope3Category.CAT_5,
        "secondary": [Scope3Category.CAT_12],
        "confidence": Decimal("0.92"),
        "description": "Waste collection, treatment and disposal activities",
    },
    "39": {
        "primary": Scope3Category.CAT_5,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Remediation activities and other waste management services",
    },
    # Construction
    "41": {
        "primary": Scope3Category.CAT_2,
        "secondary": [Scope3Category.CAT_1],
        "confidence": Decimal("0.85"),
        "description": "Construction of buildings",
    },
    "42": {
        "primary": Scope3Category.CAT_2,
        "secondary": [],
        "confidence": Decimal("0.85"),
        "description": "Civil engineering",
    },
    "43": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_2],
        "confidence": Decimal("0.82"),
        "description": "Specialized construction activities",
    },
    # Transport
    "49": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_6, Scope3Category.CAT_7],
        "confidence": Decimal("0.85"),
        "description": "Land transport and transport via pipelines",
    },
    "50": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.88"),
        "description": "Water transport",
    },
    "51": {
        "primary": Scope3Category.CAT_6,
        "secondary": [Scope3Category.CAT_4],
        "confidence": Decimal("0.90"),
        "description": "Air transport",
    },
    "52": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.85"),
        "description": "Warehousing and support activities for transportation",
    },
    "53": {
        "primary": Scope3Category.CAT_4,
        "secondary": [Scope3Category.CAT_9],
        "confidence": Decimal("0.85"),
        "description": "Postal and courier activities",
    },
    # Accommodation
    "55": {
        "primary": Scope3Category.CAT_6,
        "secondary": [],
        "confidence": Decimal("0.90"),
        "description": "Accommodation",
    },
    "56": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_14],
        "confidence": Decimal("0.80"),
        "description": "Food and beverage service activities",
    },
    # Finance
    "64": {
        "primary": Scope3Category.CAT_15,
        "secondary": [],
        "confidence": Decimal("0.92"),
        "description": "Financial service activities, except insurance and pension funding",
    },
    "65": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_15],
        "confidence": Decimal("0.82"),
        "description": "Insurance, reinsurance and pension funding",
    },
    "66": {
        "primary": Scope3Category.CAT_15,
        "secondary": [],
        "confidence": Decimal("0.88"),
        "description": "Activities auxiliary to financial service and insurance",
    },
    # Real estate
    "68": {
        "primary": Scope3Category.CAT_13,
        "secondary": [Scope3Category.CAT_8],
        "confidence": Decimal("0.88"),
        "description": "Real estate activities",
    },
    # Leasing
    "77": {
        "primary": Scope3Category.CAT_8,
        "secondary": [Scope3Category.CAT_13],
        "confidence": Decimal("0.88"),
        "description": "Rental and leasing activities",
    },
    # Waste
    "81": {
        "primary": Scope3Category.CAT_1,
        "secondary": [Scope3Category.CAT_5],
        "confidence": Decimal("0.78"),
        "description": "Services to buildings and landscape activities",
    },
}


# =============================================================================
# TABLE 3: NAICS <-> ISIC CONCORDANCE
# =============================================================================

# Maps NAICS 2-digit sector to ISIC section code
_NAICS_TO_ISIC_CONCORDANCE: Dict[str, str] = {
    "11": "A",   # Agriculture
    "21": "B",   # Mining
    "22": "D",   # Utilities (electric/gas)
    "23": "F",   # Construction
    "31": "C",   # Manufacturing
    "32": "C",   # Manufacturing
    "33": "C",   # Manufacturing
    "42": "G",   # Wholesale
    "44": "G",   # Retail
    "45": "G",   # Retail
    "48": "H",   # Transportation
    "49": "H",   # Transportation / postal
    "51": "J",   # Information
    "52": "K",   # Finance
    "53": "L",   # Real estate
    "54": "M",   # Professional services
    "55": "M",   # Management of companies
    "56": "N",   # Administrative services
    "61": "P",   # Education
    "62": "Q",   # Health care
    "71": "R",   # Arts/Entertainment
    "72": "I",   # Accommodation/Food
    "81": "S",   # Other services
    "92": "O",   # Public administration
}

# Reverse concordance: ISIC section -> NAICS 2-digit sectors
_ISIC_TO_NAICS_CONCORDANCE: Dict[str, List[str]] = {
    "A": ["11"],
    "B": ["21"],
    "C": ["31", "32", "33"],
    "D": ["22"],
    "E": ["22"],  # Water/waste utilities
    "F": ["23"],
    "G": ["42", "44", "45"],
    "H": ["48", "49"],
    "I": ["72"],
    "J": ["51"],
    "K": ["52"],
    "L": ["53"],
    "M": ["54", "55"],
    "N": ["56"],
    "O": ["92"],
    "P": ["61"],
    "Q": ["62"],
    "R": ["71"],
    "S": ["81"],
    "T": [],
    "U": [],
}


# =============================================================================
# TABLE 4: GL ACCOUNT RANGE -> SCOPE 3 CATEGORY
# =============================================================================

# Each tuple: (range_start, range_end, primary_category, secondary_categories, description)
_GL_ACCOUNT_RANGES: List[Tuple[int, int, Scope3Category, List[Scope3Category], str]] = [
    (5000, 5199, Scope3Category.CAT_1, [], "COGS - Materials"),
    (5200, 5299, Scope3Category.CAT_1, [], "COGS - Direct Labor (outsourced)"),
    (5300, 5399, Scope3Category.CAT_4, [Scope3Category.CAT_9], "Freight In"),
    (5400, 5499, Scope3Category.CAT_1, [], "Subcontractor Services"),
    (5500, 5599, Scope3Category.CAT_1, [Scope3Category.CAT_4], "Packaging Materials"),
    (6100, 6199, Scope3Category.CAT_1, [], "Office Supplies"),
    (6200, 6299, Scope3Category.CAT_1, [Scope3Category.CAT_2], "IT & Software"),
    (6300, 6399, Scope3Category.CAT_1, [], "Professional Services"),
    (6400, 6499, Scope3Category.CAT_6, [Scope3Category.CAT_7], "Travel & Entertainment"),
    (6500, 6599, Scope3Category.CAT_3, [Scope3Category.CAT_7], "Vehicle / Fleet"),
    (6600, 6699, Scope3Category.CAT_3, [], "Utilities (electric, gas, water)"),
    (6700, 6799, Scope3Category.CAT_8, [Scope3Category.CAT_13], "Rent & Leases"),
    (6800, 6899, Scope3Category.CAT_1, [], "Insurance"),
    (6900, 6999, Scope3Category.CAT_1, [], "Repairs & Maintenance"),
    (7000, 7999, Scope3Category.CAT_2, [], "Capital Expenditures"),
    (8000, 8099, Scope3Category.CAT_5, [Scope3Category.CAT_12], "Waste Disposal"),
    (8100, 8199, Scope3Category.CAT_9, [Scope3Category.CAT_4], "Distribution / Outbound Freight"),
    (8200, 8299, Scope3Category.CAT_14, [], "Franchise Fees & Royalties"),
    (8300, 8399, Scope3Category.CAT_15, [], "Investment Expenses"),
    (8400, 8499, Scope3Category.CAT_7, [], "Employee Commuting Programs"),
]


# =============================================================================
# TABLE 5: SPEND KEYWORD DICTIONARY (500+ keywords)
# =============================================================================

# Structure: keyword -> (category, confidence, group)
# Organised by keyword group for maintainability

_KEYWORD_MAPPINGS: Dict[str, Tuple[Scope3Category, Decimal, str]] = {}


def _register_keywords(
    keywords: List[str],
    category: Scope3Category,
    confidence: Decimal,
    group: str,
) -> None:
    """Register a batch of keywords into the keyword dictionary."""
    for kw in keywords:
        _KEYWORD_MAPPINGS[kw.lower()] = (category, confidence, group)


# ---- Cat 1: Purchased Goods & Services (raw_materials, components, supplies) ----
_register_keywords([
    "raw material", "raw materials", "raw material purchase", "material supply",
    "component", "components", "component parts", "sub-component", "subcomponent",
    "office supplies", "office supply", "stationery", "paper", "toner", "ink",
    "cleaning supplies", "cleaning products", "janitorial supplies", "sanitation",
    "food ingredients", "ingredients", "food supply", "beverages", "catering",
    "chemicals", "chemical supply", "reagent", "reagents", "lab supplies",
    "packaging", "packaging material", "packaging supplies", "cartons", "boxes",
    "textiles", "fabric", "cloth", "yarn", "fiber", "fibre",
    "plastic", "plastics", "polymer", "resin", "rubber",
    "metal", "metals", "steel", "aluminum", "aluminium", "copper", "iron",
    "wood", "lumber", "timber", "plywood", "veneer",
    "glass", "ceramics", "cement", "concrete", "aggregate", "sand", "gravel",
    "paint", "coating", "coatings", "adhesive", "adhesives", "sealant",
    "purchased goods", "goods purchased", "consumables", "consumable",
    "software license", "software subscription", "saas", "cloud service",
    "cloud services", "hosting", "web hosting", "domain", "license fee",
    "consulting", "consultancy", "advisory", "professional service",
    "professional services", "legal services", "legal fees", "audit services",
    "audit fees", "accounting services", "tax services", "bookkeeping",
    "marketing services", "advertising", "media services", "pr services",
    "design services", "creative services", "print services", "printing",
    "security services", "guard services", "cleaning service", "janitorial service",
    "maintenance service", "maintenance contract", "repair services",
    "it services", "managed services", "outsourced services", "bpo",
    "call center", "customer service", "help desk", "support contract",
    "staffing", "temp labor", "temporary staff", "contract labor",
    "training services", "education services", "conference fees",
    "medical supplies", "pharmaceutical", "pharmaceuticals", "drug supply",
    "uniforms", "workwear", "ppe", "personal protective equipment",
    "tools", "hand tools", "power tools", "test equipment",
    "batteries", "filters", "gaskets", "seals", "bearings",
    "electronics", "electronic components", "circuit boards", "semiconductors",
    "wiring", "cables", "connectors", "fasteners", "bolts", "screws", "nails",
    "pipes", "tubing", "fittings", "valves", "pumps", "compressors",
    "fertilizer", "pesticide", "herbicide", "seed", "seeds", "animal feed",
    "office furniture", "desk", "chair", "shelving", "storage",
    "subscription", "membership", "dues", "permit fees",
    "telecom", "telecommunications", "phone service", "internet service",
    "water supply", "bottled water",
], Scope3Category.CAT_1, Decimal("0.55"), "raw_materials_and_services")

# ---- Cat 2: Capital Goods (machinery, equipment, vehicles) ----
_register_keywords([
    "machinery", "machine", "machines", "industrial machinery", "heavy machinery",
    "equipment", "capital equipment", "production equipment", "processing equipment",
    "manufacturing equipment", "factory equipment", "plant equipment",
    "vehicle purchase", "vehicle acquisition", "fleet purchase", "truck purchase",
    "car purchase", "van purchase", "forklift", "forklift purchase",
    "building construction", "building purchase", "property purchase",
    "building renovation", "major renovation", "building expansion",
    "hvac system", "hvac installation", "hvac equipment", "boiler", "chiller",
    "generator", "backup generator", "transformer", "switchgear",
    "solar panel", "solar installation", "wind turbine", "renewable installation",
    "server", "servers", "data center equipment", "network equipment",
    "it infrastructure", "computer hardware", "laptop purchase", "desktop purchase",
    "printer purchase", "copier purchase", "scanner purchase",
    "furniture purchase", "fixture", "fixtures", "leasehold improvement",
    "capital improvement", "capital project", "capex", "capital expenditure",
    "construction project", "infrastructure project",
    "conveyor", "conveyor system", "robotic", "robotics", "automation equipment",
    "tooling", "die", "mold", "mould", "jig",
    "crane", "hoist", "winch", "elevator", "escalator",
    "security system", "camera system", "access control system",
    "laboratory equipment", "medical equipment", "diagnostic equipment",
    "refrigeration unit", "cold storage", "freezer unit",
    "tank", "storage tank", "silo", "vessel", "reactor",
    "compressor purchase", "pump purchase", "motor purchase",
    "aircraft", "aircraft purchase", "vessel purchase", "ship purchase",
], Scope3Category.CAT_2, Decimal("0.58"), "capital_goods")

# ---- Cat 3: Fuel & Energy Activities ----
_register_keywords([
    "electricity", "electric bill", "electric utility", "power bill",
    "natural gas", "gas bill", "gas utility", "propane", "lpg",
    "diesel fuel", "gasoline", "petrol", "kerosene", "jet fuel",
    "heating oil", "fuel oil", "bunker fuel", "marine fuel",
    "coal", "coke", "biomass fuel", "wood pellet", "wood chip",
    "energy purchase", "energy supply", "power purchase", "power supply",
    "steam purchase", "steam supply", "district heating", "chilled water",
    "cooling purchase", "cooling supply",
    "renewable energy certificate", "rec", "recs", "green certificate",
    "power purchase agreement", "ppa", "virtual ppa",
    "grid electricity", "utility bill", "energy invoice",
    "fuel purchase", "fuel supply", "fuel delivery", "fuel card",
    "charging station", "ev charging", "electric vehicle charging",
    "hydrogen", "hydrogen fuel", "green hydrogen",
    "biogas", "biodiesel", "ethanol", "biofuel", "renewable fuel",
], Scope3Category.CAT_3, Decimal("0.60"), "fuel_and_energy")

# ---- Cat 4: Upstream Transportation & Distribution ----
_register_keywords([
    "freight", "freight in", "inbound freight", "freight charge", "freight cost",
    "shipping", "shipping cost", "shipping charge", "shipping fee",
    "logistics", "logistics service", "logistics provider", "3pl",
    "trucking", "trucking service", "ltl", "ftl", "truckload",
    "rail freight", "rail transport", "rail shipment", "intermodal",
    "ocean freight", "sea freight", "container shipping", "fcl", "lcl",
    "air freight", "air cargo", "air shipment", "express delivery",
    "courier", "courier service", "parcel delivery", "package delivery",
    "warehousing", "warehouse", "warehouse storage", "storage fee",
    "distribution", "distribution service", "cross-dock", "cross docking",
    "customs", "customs clearance", "customs duty", "import duty",
    "freight forwarder", "freight forwarding", "broker", "customs broker",
    "pallet", "palletization", "crating", "drayage",
    "cold chain", "cold storage", "refrigerated transport", "reefer",
    "carrier", "carrier service", "haulage", "haulier", "transport service",
    "supply chain", "supply chain logistics", "inbound logistics",
    "delivery service", "last mile delivery", "last-mile",
], Scope3Category.CAT_4, Decimal("0.58"), "upstream_transport")

# ---- Cat 5: Waste Generated in Operations ----
_register_keywords([
    "waste disposal", "waste management", "waste collection", "waste removal",
    "waste treatment", "waste processing", "waste handling",
    "landfill", "landfill disposal", "landfill fee", "tipping fee",
    "incineration", "waste incineration", "waste-to-energy",
    "recycling", "recycling service", "recycling fee", "recyclable",
    "composting", "compost", "organic waste", "food waste",
    "hazardous waste", "hazmat disposal", "chemical waste", "toxic waste",
    "medical waste", "clinical waste", "biohazard waste",
    "electronic waste", "e-waste", "ewaste", "it asset disposal",
    "wastewater", "wastewater treatment", "effluent treatment",
    "sewage", "sewage treatment", "sewage disposal",
    "scrap", "scrap metal", "scrap disposal", "metal recycling",
    "demolition waste", "construction waste", "demolition debris",
    "skip hire", "dumpster", "dumpster rental", "bin hire",
    "waste manifest", "waste transfer", "waste transfer note",
    "shredding", "document shredding", "secure destruction",
    "oil disposal", "oil recycling", "grease trap", "fat trap",
    "waste audit", "waste assessment", "waste characterization",
], Scope3Category.CAT_5, Decimal("0.60"), "waste_disposal")

# ---- Cat 6: Business Travel ----
_register_keywords([
    "air travel", "airfare", "airline ticket", "flight", "flights",
    "flight booking", "airline booking", "air ticket", "boarding pass",
    "business class", "first class", "economy class", "cabin upgrade",
    "hotel", "hotel room", "hotel booking", "hotel stay", "accommodation",
    "motel", "lodging", "room night", "room rate", "hotel expense",
    "car rental", "rental car", "vehicle rental", "hire car",
    "taxi", "taxi fare", "uber", "lyft", "ride-hailing", "rideshare",
    "train ticket", "rail ticket", "rail travel", "rail fare", "rail pass",
    "bus ticket", "coach ticket", "bus fare",
    "travel expense", "travel reimbursement", "travel per diem", "per diem",
    "business trip", "business travel", "corporate travel",
    "conference travel", "meeting travel", "client visit travel",
    "travel agency", "travel management", "travel booking", "tmc",
    "travel insurance", "trip insurance",
    "ferry ticket", "ferry booking", "ferry fare",
    "mileage reimbursement", "mileage claim", "km reimbursement",
    "travel meal", "meal expense", "meal allowance",
    "airport transfer", "ground transportation", "airport shuttle",
    "visa fee", "passport fee", "travel visa",
    "baggage fee", "excess baggage", "luggage fee",
    "parking", "parking fee", "valet parking", "airport parking",
], Scope3Category.CAT_6, Decimal("0.60"), "business_travel")

# ---- Cat 7: Employee Commuting ----
_register_keywords([
    "commuting", "commute", "commuter", "commuter benefit", "commuter program",
    "shuttle", "shuttle bus", "shuttle service", "employee shuttle",
    "transit pass", "transit subsidy", "bus pass", "metro pass", "subway pass",
    "bike program", "bicycle program", "cycle to work", "bike-to-work",
    "carpool", "car pool", "carpooling", "vanpool", "van pool",
    "remote work", "telework", "work from home", "wfh",
    "telecommuting", "home office", "home office stipend",
    "parking subsidy", "employee parking", "parking permit",
    "ev subsidy", "electric vehicle subsidy", "charging benefit",
    "commuter rail", "commuter bus", "commuter ferry",
], Scope3Category.CAT_7, Decimal("0.55"), "commuting")

# ---- Cat 8: Upstream Leased Assets ----
_register_keywords([
    "office lease", "office rent", "building lease", "building rent",
    "warehouse lease", "warehouse rent", "factory lease",
    "equipment lease", "equipment rental", "machinery lease", "machinery rental",
    "vehicle lease", "vehicle rental", "fleet lease", "car lease",
    "it lease", "computer lease", "server lease", "copier lease",
    "printer lease", "furniture lease", "furniture rental",
    "lease payment", "rent payment", "lease expense", "rental expense",
    "operating lease", "finance lease", "capital lease",
    "co-working", "coworking", "shared office", "serviced office",
    "storage lease", "storage rental", "land lease", "land rent",
], Scope3Category.CAT_8, Decimal("0.55"), "upstream_leases")

# ---- Cat 9: Downstream Transportation & Distribution ----
_register_keywords([
    "outbound freight", "outbound shipping", "outbound logistics",
    "distribution", "distribution service", "product distribution",
    "delivery to customer", "customer delivery", "last mile",
    "outbound transport", "outbound delivery", "shipping to customer",
    "product shipping", "product delivery", "order fulfillment",
    "distribution center", "fulfillment center", "dc operations",
    "downstream logistics", "downstream transport", "downstream freight",
    "white glove delivery", "installation delivery", "home delivery",
    "reverse logistics", "returns processing", "return shipment",
    "third-party distribution", "3pl outbound", "outsourced distribution",
], Scope3Category.CAT_9, Decimal("0.55"), "downstream_transport")

# ---- Cat 10: Processing of Sold Products ----
_register_keywords([
    "processing of sold products", "intermediate product", "intermediate goods",
    "semi-finished", "semi-finished goods", "work in progress",
    "downstream processing", "further processing", "customer processing",
    "component sold", "part sold", "ingredient sold", "raw material sold",
    "tolling", "toll manufacturing", "toll processing", "contract manufacturing",
    "co-manufacturing", "co-packing", "private label",
], Scope3Category.CAT_10, Decimal("0.50"), "processing_sold_products")

# ---- Cat 11: Use of Sold Products ----
_register_keywords([
    "product use", "product usage", "use of sold products",
    "product energy", "product electricity", "energy consuming product",
    "fuel-consuming product", "fuel consuming", "combustion product",
    "vehicle sold", "automobile sold", "car sold",
    "appliance sold", "device sold", "equipment sold",
    "direct use emission", "indirect use emission",
    "product lifecycle", "product lifetime", "product lifespan",
    "consumer use", "end-user use", "customer use",
], Scope3Category.CAT_11, Decimal("0.50"), "use_of_sold_products")

# ---- Cat 12: End-of-Life Treatment of Sold Products ----
_register_keywords([
    "end of life", "end-of-life", "eol", "eol treatment",
    "product disposal", "product recycling", "product end of life",
    "take-back", "takeback", "take back program",
    "product recovery", "material recovery", "circularity",
    "extended producer responsibility", "epr", "weee",
    "packaging end of life", "packaging disposal", "packaging recycling",
    "product waste", "post-consumer waste", "consumer waste",
], Scope3Category.CAT_12, Decimal("0.50"), "end_of_life")

# ---- Cat 13: Downstream Leased Assets ----
_register_keywords([
    "leased to tenant", "tenant lease", "tenant rent",
    "rental income", "lease income", "property lease out",
    "lessor", "lessor obligation", "asset leased out",
    "sublease", "sub-lease", "subletting",
    "investment property", "rental property", "leased property",
    "equipment leased out", "vehicle leased out",
    "operating lease income", "finance lease income",
], Scope3Category.CAT_13, Decimal("0.52"), "downstream_leases")

# ---- Cat 14: Franchises ----
_register_keywords([
    "franchise", "franchise fee", "franchise royalty",
    "franchise agreement", "franchise operation", "franchise unit",
    "franchisor", "franchisee", "franchise network",
    "franchise revenue", "franchise payment", "franchise license",
    "brand license", "brand licensing", "brand fee",
    "franchise marketing", "franchise advertising", "co-op advertising",
    "franchise support", "franchise training",
    "master franchise", "sub-franchise", "area developer",
], Scope3Category.CAT_14, Decimal("0.60"), "franchise")

# ---- Cat 15: Investments ----
_register_keywords([
    "investment", "investments", "investment portfolio",
    "equity investment", "stock investment", "share purchase",
    "bond investment", "bond purchase", "fixed income",
    "private equity", "venture capital", "vc investment",
    "project finance", "infrastructure investment", "real estate investment",
    "reit", "real estate fund", "property fund",
    "mortgage portfolio", "mortgage lending", "home loan",
    "motor vehicle loan", "auto loan", "vehicle finance",
    "sovereign bond", "government bond", "treasury bond",
    "mutual fund", "index fund", "etf", "exchange traded fund",
    "portfolio", "portfolio management", "asset management",
    "fund management", "wealth management", "investment management",
    "financed emissions", "financed emission", "pcaf",
    "credit facility", "loan portfolio", "lending portfolio",
    "green bond", "sustainability bond", "social bond",
    "impact investment", "esg investment", "sustainable investment",
], Scope3Category.CAT_15, Decimal("0.58"), "investments")


# =============================================================================
# TABLE 6: CATEGORY INFO CONSTANTS (15 categories)
# =============================================================================

_CATEGORY_INFO: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Purchased Goods and Services",
        "description": (
            "Extraction, production, and transportation of goods and services "
            "purchased or acquired by the reporting company in the reporting year, "
            "not otherwise included in Categories 2-8."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 1 (Scope 3 Standard)",
        "reporter_role": "Buyer of goods and services",
        "typical_data_sources": [
            "Accounts payable ledger", "Procurement spend data",
            "Supplier invoices", "Purchase orders",
            "Supplier-specific emission data", "EEIO databases",
        ],
        "downstream_agent": "AGENT-MRV-014",
    },
    2: {
        "name": "Capital Goods",
        "description": (
            "Extraction, production, and transportation of capital goods purchased "
            "or acquired by the reporting company in the reporting year. Capital "
            "goods are final products with an extended life that are used by the "
            "company to manufacture, deliver, or sell products/services."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 2 (Scope 3 Standard)",
        "reporter_role": "Buyer of capital goods",
        "typical_data_sources": [
            "Fixed asset register", "Capital expenditure records",
            "Equipment purchase orders", "Construction invoices",
        ],
        "downstream_agent": "AGENT-MRV-015",
    },
    3: {
        "name": "Fuel- and Energy-Related Activities",
        "description": (
            "Extraction, production, and transportation of fuels and energy "
            "purchased or acquired by the reporting company in the reporting year, "
            "not already accounted for in Scope 1 or Scope 2. Includes upstream "
            "emissions of purchased fuels, upstream emissions of purchased "
            "electricity, and transmission and distribution losses."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 3 (Scope 3 Standard)",
        "reporter_role": "Consumer of fuels and energy",
        "typical_data_sources": [
            "Energy invoices", "Fuel purchase records",
            "Utility bills", "Grid emission factor databases",
        ],
        "downstream_agent": "AGENT-MRV-016",
    },
    4: {
        "name": "Upstream Transportation and Distribution",
        "description": (
            "Transportation and distribution of products purchased by the "
            "reporting company in the reporting year between the company's "
            "tier 1 suppliers and its own operations, in vehicles and facilities "
            "not owned or controlled by the reporting company. Also includes "
            "third-party transportation and distribution services purchased by "
            "the reporting company."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 4 (Scope 3 Standard)",
        "reporter_role": "Buyer paying for freight (based on Incoterms)",
        "typical_data_sources": [
            "Freight invoices", "Shipping records",
            "Logistics provider data", "Bill of lading",
        ],
        "downstream_agent": "AGENT-MRV-017",
    },
    5: {
        "name": "Waste Generated in Operations",
        "description": (
            "Disposal and treatment of waste generated in the reporting "
            "company's operations in the reporting year, in facilities not "
            "owned or controlled by the reporting company."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 5 (Scope 3 Standard)",
        "reporter_role": "Generator of waste",
        "typical_data_sources": [
            "Waste manifests", "Waste transfer notes",
            "Waste contractor invoices", "Environmental compliance records",
        ],
        "downstream_agent": "AGENT-MRV-018",
    },
    6: {
        "name": "Business Travel",
        "description": (
            "Transportation of employees for business-related activities "
            "during the reporting year, in vehicles not owned or operated "
            "by the reporting company."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 6 (Scope 3 Standard)",
        "reporter_role": "Employer",
        "typical_data_sources": [
            "Travel booking systems", "Expense reports",
            "Travel management company data", "Corporate credit card data",
        ],
        "downstream_agent": "AGENT-MRV-019",
    },
    7: {
        "name": "Employee Commuting",
        "description": (
            "Transportation of employees between their homes and their "
            "worksites during the reporting year, in vehicles not owned or "
            "operated by the reporting company. This includes emissions from "
            "teleworking (working from home)."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 7 (Scope 3 Standard)",
        "reporter_role": "Employer",
        "typical_data_sources": [
            "Employee commuting surveys", "HR records",
            "Transit pass data", "Parking records",
        ],
        "downstream_agent": "AGENT-MRV-020",
    },
    8: {
        "name": "Upstream Leased Assets",
        "description": (
            "Operation of assets leased by the reporting company (lessee) "
            "in the reporting year, not included in Scope 1 and Scope 2. "
            "Relevant only when the reporting company is a lessee."
        ),
        "direction": ValueChainDirection.UPSTREAM,
        "ghg_protocol_chapter": "Chapter 8 (Scope 3 Standard)",
        "reporter_role": "Lessee (tenant)",
        "typical_data_sources": [
            "Lease agreements", "Landlord utility data",
            "Building management system data", "Equipment specifications",
        ],
        "downstream_agent": "AGENT-MRV-021",
    },
    9: {
        "name": "Downstream Transportation and Distribution",
        "description": (
            "Transportation and distribution of products sold by the reporting "
            "company in the reporting year between the company's operations "
            "and the end consumer, in vehicles and facilities not owned or "
            "controlled by the reporting company."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 9 (Scope 3 Standard)",
        "reporter_role": "Seller of goods",
        "typical_data_sources": [
            "Distribution records", "Outbound freight data",
            "Customer delivery records", "Logistics partner data",
        ],
        "downstream_agent": "AGENT-MRV-022",
    },
    10: {
        "name": "Processing of Sold Products",
        "description": (
            "Processing of intermediate products sold by the reporting company "
            "in the reporting year by third parties subsequent to sale. "
            "Applicable when the reporting company sells intermediate products "
            "that require further processing before use."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 10 (Scope 3 Standard)",
        "reporter_role": "Seller of intermediate products",
        "typical_data_sources": [
            "Product specifications", "Customer processing data",
            "Industry average processing data", "BOM analysis",
        ],
        "downstream_agent": "AGENT-MRV-023",
    },
    11: {
        "name": "Use of Sold Products",
        "description": (
            "End use of goods and services sold by the reporting company "
            "in the reporting year. Includes both direct use-phase emissions "
            "(from products that directly consume energy or produce emissions) "
            "and indirect use-phase emissions (from products that indirectly "
            "consume energy during use)."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 11 (Scope 3 Standard)",
        "reporter_role": "Seller of energy-consuming or emission-producing products",
        "typical_data_sources": [
            "Product energy ratings", "Product specifications",
            "Expected product lifetime data", "Industry use-phase data",
        ],
        "downstream_agent": "AGENT-MRV-024",
    },
    12: {
        "name": "End-of-Life Treatment of Sold Products",
        "description": (
            "Waste disposal and treatment of products sold by the reporting "
            "company at the end of their life. Includes the total expected "
            "end-of-life emissions of all products sold in the reporting year."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 12 (Scope 3 Standard)",
        "reporter_role": "Seller of products",
        "typical_data_sources": [
            "Product material composition", "Waste treatment statistics",
            "Regional recycling rates", "Product take-back data",
        ],
        "downstream_agent": "AGENT-MRV-025",
    },
    13: {
        "name": "Downstream Leased Assets",
        "description": (
            "Operation of assets owned by the reporting company (lessor) "
            "and leased to other entities in the reporting year, not "
            "included in Scope 1 and Scope 2. Relevant only when the "
            "reporting company is a lessor."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 13 (Scope 3 Standard)",
        "reporter_role": "Lessor (asset owner)",
        "typical_data_sources": [
            "Lease agreements", "Tenant energy data",
            "Building certifications", "Asset register",
        ],
        "downstream_agent": "AGENT-MRV-026",
    },
    14: {
        "name": "Franchises",
        "description": (
            "Operation of franchises not included in Scope 1 and Scope 2. "
            "A franchise is a business operating under a license to sell or "
            "distribute another company's goods or services within a certain "
            "location. This category is relevant to franchisors."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 14 (Scope 3 Standard)",
        "reporter_role": "Franchisor",
        "typical_data_sources": [
            "Franchise agreements", "Franchisee energy data",
            "Franchise unit EUI benchmarks", "Revenue data",
        ],
        "downstream_agent": "AGENT-MRV-027",
    },
    15: {
        "name": "Investments",
        "description": (
            "Operation of investments (including equity and debt investments "
            "and project finance) not included in Scope 1 and Scope 2. "
            "Category 15 is relevant to investors and companies that provide "
            "financial services."
        ),
        "direction": ValueChainDirection.DOWNSTREAM,
        "ghg_protocol_chapter": "Chapter 15 (Scope 3 Standard)",
        "reporter_role": "Investor or financial institution",
        "typical_data_sources": [
            "Investment portfolio data", "PCAF data",
            "Investee company emissions data", "Sovereign emissions databases",
        ],
        "downstream_agent": "AGENT-MRV-028",
    },
}


# =============================================================================
# PROVENANCE HELPER
# =============================================================================


def _calculate_hash(*parts: Any) -> str:
    """
    Calculate a SHA-256 provenance hash from variable inputs.

    Args:
        *parts: Variable number of input values to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for part in parts:
        if isinstance(part, Decimal):
            hash_input += str(part.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        elif isinstance(part, BaseModel):
            hash_input += json.dumps(
                part.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(part, (list, dict)):
            hash_input += json.dumps(part, sort_keys=True, default=str)
        elif isinstance(part, Enum):
            hash_input += str(part.value)
        else:
            hash_input += str(part)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# =============================================================================
# ENGINE CLASS
# =============================================================================


class CategoryDatabaseEngine:
    """
    Thread-safe singleton engine for classification code lookups.

    Provides deterministic, zero-hallucination lookups across NAICS 2022,
    ISIC Rev 4, GL account ranges, and a 500+ keyword spend dictionary.
    Every lookup returns a provenance hash for audit trail integrity.

    This engine does NOT perform any LLM or ML calls. All mappings are
    retrieved from frozen constant tables defined in this module.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        _lookup_count: Total number of lookups performed across all methods.

    Example:
        >>> engine = CategoryDatabaseEngine()
        >>> naics_result = engine.lookup_naics("481")
        >>> naics_result.primary_category
        <Scope3Category.CAT_6: 6>
        >>> gl_result = engine.lookup_gl_account("6400")
        >>> gl_result.primary_category
        <Scope3Category.CAT_6: 6>
    """

    _instance: Optional["CategoryDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CategoryDatabaseEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the database engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._lookup_count: int = 0
        self._lookup_lock: threading.Lock = threading.Lock()

        # Pre-compile keyword patterns for efficient matching
        self._sorted_keywords: List[Tuple[str, Scope3Category, Decimal, str]] = sorted(
            [
                (kw, cat, conf, grp)
                for kw, (cat, conf, grp) in _KEYWORD_MAPPINGS.items()
            ],
            key=lambda x: len(x[0]),
            reverse=True,  # Longest match first
        )

        logger.info(
            "CategoryDatabaseEngine initialized: "
            "naics_sectors=%d, naics_subsectors=%d, "
            "isic_sections=%d, isic_divisions=%d, "
            "gl_ranges=%d, keywords=%d, categories=%d",
            len(_NAICS_SECTOR_MAPPINGS),
            len(_NAICS_SUBSECTOR_OVERRIDES),
            len(_ISIC_SECTION_MAPPINGS),
            len(_ISIC_DIVISION_OVERRIDES),
            len(_GL_ACCOUNT_RANGES),
            len(_KEYWORD_MAPPINGS),
            len(_CATEGORY_INFO),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to 8 decimal places."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    # =========================================================================
    # NAICS LOOKUPS
    # =========================================================================

    def lookup_naics(self, code: str) -> NAICSLookupResult:
        """
        Look up a NAICS code and return the primary Scope 3 category mapping.

        Uses a hierarchical resolution strategy:
        1. Try 3-digit subsector override (highest specificity)
        2. Fall back to 2-digit sector mapping (broad mapping)

        Supports 2-digit through 6-digit NAICS codes. The code is matched
        by progressively truncating from left until a match is found.

        Args:
            code: NAICS code string (2-6 digits).

        Returns:
            NAICSLookupResult with primary category, confidence, and provenance hash.

        Raises:
            ValueError: If the NAICS code is empty or no mapping is found.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> result = engine.lookup_naics("484110")
            >>> result.primary_category
            <Scope3Category.CAT_4: 4>
            >>> result.description
            'Truck Transportation'
        """
        self._increment_lookup()
        start_time = datetime.now(timezone.utc)

        code = code.strip()
        if not code or not code.isdigit():
            raise ValueError(
                f"Invalid NAICS code '{code}'. Must be a non-empty numeric string."
            )

        # Try progressively shorter prefixes: 3-digit then 2-digit
        matched_code: Optional[str] = None
        mapping: Optional[Dict[str, Any]] = None

        # 1. Try 3-digit subsector override
        prefix_3 = code[:3] if len(code) >= 3 else None
        if prefix_3 and prefix_3 in _NAICS_SUBSECTOR_OVERRIDES:
            matched_code = prefix_3
            mapping = _NAICS_SUBSECTOR_OVERRIDES[prefix_3]

        # 2. Fall back to 2-digit sector
        if mapping is None:
            prefix_2 = code[:2]
            if prefix_2 in _NAICS_SECTOR_MAPPINGS:
                matched_code = prefix_2
                mapping = _NAICS_SECTOR_MAPPINGS[prefix_2]

        if mapping is None or matched_code is None:
            raise ValueError(
                f"No NAICS mapping found for code '{code}'. "
                f"Available 2-digit sectors: {sorted(_NAICS_SECTOR_MAPPINGS.keys())}"
            )

        provenance_hash = _calculate_hash(
            "naics_lookup", code, matched_code,
            mapping["primary"], mapping["confidence"],
        )

        result = NAICSLookupResult(
            naics_code=code,
            matched_code=matched_code,
            primary_category=mapping["primary"],
            secondary_categories=mapping["secondary"],
            confidence=self._quantize(mapping["confidence"]),
            description=mapping["description"],
            provenance_hash=provenance_hash,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.debug(
            "NAICS lookup: code=%s, matched=%s, category=Cat %d, "
            "confidence=%s, elapsed_ms=%.2f",
            code, matched_code, mapping["primary"].value,
            mapping["confidence"], elapsed_ms,
        )

        return result

    # =========================================================================
    # ISIC LOOKUPS
    # =========================================================================

    def lookup_isic(self, code: str) -> ISICLookupResult:
        """
        Look up an ISIC Rev 4 code and return the primary Scope 3 category mapping.

        Uses a hierarchical resolution strategy:
        1. Try 2-digit division override (highest specificity)
        2. Fall back to 1-character section mapping (broad mapping)

        Args:
            code: ISIC code string (1-character section or 2-digit division).

        Returns:
            ISICLookupResult with primary category, confidence, and provenance hash.

        Raises:
            ValueError: If the ISIC code is empty or no mapping is found.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> result = engine.lookup_isic("51")
            >>> result.primary_category
            <Scope3Category.CAT_6: 6>
            >>> result.description
            'Air transport'
        """
        self._increment_lookup()
        start_time = datetime.now(timezone.utc)

        code = code.strip().upper()
        if not code:
            raise ValueError("ISIC code cannot be empty.")

        matched_code: Optional[str] = None
        mapping: Optional[Dict[str, Any]] = None

        # 1. Try 2-digit division override (numeric codes)
        numeric_part = code.lstrip("ABCDEFGHIJKLMNOPQRSTU")
        if len(numeric_part) >= 2:
            div_2 = numeric_part[:2]
            if div_2 in _ISIC_DIVISION_OVERRIDES:
                matched_code = div_2
                mapping = _ISIC_DIVISION_OVERRIDES[div_2]

        # 2. Fall back to section letter
        if mapping is None:
            section = code[0].upper()
            if section in _ISIC_SECTION_MAPPINGS:
                matched_code = section
                mapping = _ISIC_SECTION_MAPPINGS[section]

        if mapping is None or matched_code is None:
            raise ValueError(
                f"No ISIC mapping found for code '{code}'. "
                f"Available sections: {sorted(_ISIC_SECTION_MAPPINGS.keys())}"
            )

        provenance_hash = _calculate_hash(
            "isic_lookup", code, matched_code,
            mapping["primary"], mapping["confidence"],
        )

        result = ISICLookupResult(
            isic_code=code,
            matched_code=matched_code,
            primary_category=mapping["primary"],
            secondary_categories=mapping["secondary"],
            confidence=self._quantize(mapping["confidence"]),
            description=mapping["description"],
            provenance_hash=provenance_hash,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.debug(
            "ISIC lookup: code=%s, matched=%s, category=Cat %d, "
            "confidence=%s, elapsed_ms=%.2f",
            code, matched_code, mapping["primary"].value,
            mapping["confidence"], elapsed_ms,
        )

        return result

    # =========================================================================
    # GL ACCOUNT LOOKUPS
    # =========================================================================

    def lookup_gl_account(self, account_code: str) -> GLLookupResult:
        """
        Look up a GL account code and return the Scope 3 category mapping.

        Scans the GL account range table to find the range that contains the
        given account code. Returns the first matching range.

        Args:
            account_code: GL account code (numeric string, e.g. "6410").

        Returns:
            GLLookupResult with primary category, confidence, and provenance hash.

        Raises:
            ValueError: If the account code is not numeric or no mapping is found.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> result = engine.lookup_gl_account("6450")
            >>> result.primary_category
            <Scope3Category.CAT_6: 6>
            >>> result.matched_range
            '6400-6499'
        """
        self._increment_lookup()
        start_time = datetime.now(timezone.utc)

        code_str = account_code.strip()
        if not code_str.isdigit():
            raise ValueError(
                f"Invalid GL account code '{account_code}'. Must be a numeric string."
            )

        code_int = int(code_str)

        for range_start, range_end, primary_cat, secondary_cats, desc in _GL_ACCOUNT_RANGES:
            if range_start <= code_int <= range_end:
                matched_range = f"{range_start}-{range_end}"
                confidence = Decimal("0.85")

                provenance_hash = _calculate_hash(
                    "gl_account_lookup", account_code,
                    matched_range, primary_cat, confidence,
                )

                result = GLLookupResult(
                    account_code=account_code,
                    matched_range=matched_range,
                    primary_category=primary_cat,
                    secondary_categories=secondary_cats,
                    confidence=self._quantize(confidence),
                    description=desc,
                    provenance_hash=provenance_hash,
                )

                elapsed_ms = (
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
                logger.debug(
                    "GL account lookup: code=%s, range=%s, category=Cat %d, "
                    "elapsed_ms=%.2f",
                    account_code, matched_range, primary_cat.value, elapsed_ms,
                )

                return result

        raise ValueError(
            f"No GL account mapping found for code '{account_code}'. "
            f"Available ranges: "
            + ", ".join(
                f"{s}-{e}" for s, e, _, _, _ in _GL_ACCOUNT_RANGES
            )
        )

    # =========================================================================
    # KEYWORD LOOKUPS
    # =========================================================================

    def lookup_keyword(self, text: str) -> KeywordLookupResult:
        """
        Look up the best matching keyword in the spend keyword dictionary.

        Performs case-insensitive substring matching against the 500+ keyword
        dictionary, selecting the longest matching keyword (most specific).
        If multiple keywords match with the same length, the one with the
        highest confidence is selected.

        Args:
            text: Input text to classify (e.g., a transaction description).

        Returns:
            KeywordLookupResult with primary category, confidence, and
            provenance hash.

        Raises:
            ValueError: If text is empty or no keyword match is found.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> result = engine.lookup_keyword("air travel booking")
            >>> result.primary_category
            <Scope3Category.CAT_6: 6>
            >>> result.keyword_group
            'business_travel'
        """
        self._increment_lookup()
        start_time = datetime.now(timezone.utc)

        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty for keyword lookup.")

        text_lower = text.lower()
        best_match: Optional[Tuple[str, Scope3Category, Decimal, str]] = None

        # Sorted by length descending -- first match is the longest
        for kw, cat, conf, grp in self._sorted_keywords:
            if kw in text_lower:
                if best_match is None or len(kw) > len(best_match[0]):
                    best_match = (kw, cat, conf, grp)
                elif len(kw) == len(best_match[0]) and conf > best_match[2]:
                    best_match = (kw, cat, conf, grp)

        if best_match is None:
            raise ValueError(
                f"No keyword match found for text '{text[:100]}'. "
                f"Total keywords available: {len(_KEYWORD_MAPPINGS)}"
            )

        matched_kw, category, confidence, group = best_match

        provenance_hash = _calculate_hash(
            "keyword_lookup", text, matched_kw, category, confidence,
        )

        result = KeywordLookupResult(
            input_text=text,
            matched_keyword=matched_kw,
            primary_category=category,
            confidence=self._quantize(confidence),
            keyword_group=group,
            provenance_hash=provenance_hash,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.debug(
            "Keyword lookup: text='%s', keyword='%s', category=Cat %d, "
            "confidence=%s, group=%s, elapsed_ms=%.2f",
            text[:50], matched_kw, category.value,
            confidence, group, elapsed_ms,
        )

        return result

    def lookup_keywords_batch(
        self, texts: List[str]
    ) -> List[KeywordLookupResult]:
        """
        Batch keyword lookup for multiple text inputs.

        Processes each text individually using lookup_keyword(). Records
        that fail to match return None in the result list position is
        skipped; only successful matches are included. For failed lookups
        a warning is logged and the text is skipped.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of KeywordLookupResult objects for successful matches.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> results = engine.lookup_keywords_batch([
            ...     "air travel expense",
            ...     "raw material purchase",
            ...     "waste disposal fee",
            ... ])
            >>> len(results)
            3
        """
        start_time = datetime.now(timezone.utc)
        results: List[KeywordLookupResult] = []

        for text in texts:
            try:
                result = self.lookup_keyword(text)
                results.append(result)
            except ValueError as exc:
                logger.warning(
                    "Keyword batch: skipped text '%s': %s",
                    text[:50], str(exc),
                )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Keyword batch lookup: total=%d, matched=%d, elapsed_ms=%.2f",
            len(texts), len(results), elapsed_ms,
        )

        return results

    # =========================================================================
    # NAICS <-> ISIC CROSS-REFERENCE
    # =========================================================================

    def get_naics_to_isic(self, naics_code: str) -> Optional[str]:
        """
        Get the ISIC Rev 4 section code corresponding to a NAICS 2-digit sector.

        Uses the NAICS-ISIC concordance table for cross-reference between
        North American and international classification systems.

        Args:
            naics_code: NAICS code (at least 2 digits).

        Returns:
            ISIC section letter (A-U) or None if no concordance exists.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> engine.get_naics_to_isic("48")
            'H'
            >>> engine.get_naics_to_isic("52")
            'K'
        """
        self._increment_lookup()

        code = naics_code.strip()
        if not code or not code.isdigit():
            logger.warning(
                "Invalid NAICS code for ISIC cross-reference: '%s'", naics_code
            )
            return None

        prefix_2 = code[:2]
        isic_section = _NAICS_TO_ISIC_CONCORDANCE.get(prefix_2)

        if isic_section is None:
            logger.debug(
                "No NAICS-ISIC concordance for sector '%s'", prefix_2
            )

        return isic_section

    def get_isic_to_naics(self, isic_code: str) -> List[str]:
        """
        Get the NAICS 2-digit sector codes corresponding to an ISIC section.

        Args:
            isic_code: ISIC section letter (A-U).

        Returns:
            List of NAICS 2-digit sector codes. Empty if no concordance exists.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> engine.get_isic_to_naics("C")
            ['31', '32', '33']
        """
        self._increment_lookup()

        section = isic_code.strip().upper()
        if not section or len(section) < 1:
            return []

        section_letter = section[0]
        return _ISIC_TO_NAICS_CONCORDANCE.get(section_letter, [])

    # =========================================================================
    # CATEGORY INFO
    # =========================================================================

    def get_category_info(self, category_number: int) -> CategoryInfo:
        """
        Get comprehensive metadata for a single Scope 3 category.

        Args:
            category_number: Category number (1-15).

        Returns:
            CategoryInfo with name, description, direction, chapter reference,
            reporter role, typical data sources, and downstream agent ID.

        Raises:
            ValueError: If category_number is not between 1 and 15.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> info = engine.get_category_info(6)
            >>> info.name
            'Business Travel'
            >>> info.direction
            <ValueChainDirection.UPSTREAM: 'upstream'>
        """
        self._increment_lookup()

        if category_number < 1 or category_number > 15:
            raise ValueError(
                f"Invalid category number {category_number}. Must be 1-15."
            )

        info_data = _CATEGORY_INFO[category_number]

        return CategoryInfo(
            number=category_number,
            name=info_data["name"],
            description=info_data["description"],
            direction=info_data["direction"],
            ghg_protocol_chapter=info_data["ghg_protocol_chapter"],
            reporter_role=info_data["reporter_role"],
            typical_data_sources=info_data["typical_data_sources"],
            downstream_agent=info_data["downstream_agent"],
        )

    def get_all_categories(self) -> List[CategoryInfo]:
        """
        Get metadata for all 15 Scope 3 categories.

        Returns:
            List of 15 CategoryInfo objects, ordered by category number.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> categories = engine.get_all_categories()
            >>> len(categories)
            15
            >>> categories[0].number
            1
        """
        return [self.get_category_info(i) for i in range(1, 16)]

    # =========================================================================
    # VERSION AND SUMMARY
    # =========================================================================

    def get_mapping_version(self) -> str:
        """
        Get the version string for all mapping tables.

        The mapping version tracks the revision of NAICS, ISIC, GL, and
        keyword tables. It follows the format YYYY.MAJOR.MINOR.

        Returns:
            Mapping version string.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> engine.get_mapping_version()
            '2026.1.0'
        """
        return MAPPING_VERSION

    def get_lookup_count(self) -> int:
        """
        Get the total number of lookups performed across all methods.

        Returns:
            Integer count of lookups.
        """
        with self._lookup_lock:
            return self._lookup_count

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database contents and statistics.

        Returns:
            Dict with counts of all mapping tables and total lookups.

        Example:
            >>> engine = CategoryDatabaseEngine()
            >>> summary = engine.get_database_summary()
            >>> summary["keyword_count"]
            500
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "mapping_version": MAPPING_VERSION,
            "naics_sector_count": len(_NAICS_SECTOR_MAPPINGS),
            "naics_subsector_count": len(_NAICS_SUBSECTOR_OVERRIDES),
            "isic_section_count": len(_ISIC_SECTION_MAPPINGS),
            "isic_division_count": len(_ISIC_DIVISION_OVERRIDES),
            "naics_isic_concordance_entries": len(_NAICS_TO_ISIC_CONCORDANCE),
            "gl_account_ranges": len(_GL_ACCOUNT_RANGES),
            "keyword_count": len(_KEYWORD_MAPPINGS),
            "category_count": len(_CATEGORY_INFO),
            "total_lookups": self.get_lookup_count(),
        }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for use in test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_engine_instance: Optional[CategoryDatabaseEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_category_database_engine() -> CategoryDatabaseEngine:
    """
    Get the singleton CategoryDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance.

    Returns:
        CategoryDatabaseEngine singleton instance.

    Example:
        >>> engine = get_category_database_engine()
        >>> result = engine.lookup_naics("484")
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = CategoryDatabaseEngine()
        return _engine_instance


def reset_category_database_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    CategoryDatabaseEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Agent metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "MAPPING_VERSION",
    # Enumerations
    "Scope3Category",
    "ValueChainDirection",
    # Result models
    "NAICSLookupResult",
    "ISICLookupResult",
    "GLLookupResult",
    "KeywordLookupResult",
    "CategoryInfo",
    # Engine class
    "CategoryDatabaseEngine",
    "get_category_database_engine",
    "reset_category_database_engine",
]
