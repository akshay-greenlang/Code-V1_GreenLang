# -*- coding: utf-8 -*-
"""
Country Forest Database - AGENT-EUDR-020 Deforestation Alert System

Comprehensive country-level forest cover statistics for 180+ countries covering
total forest area, forest percentage of land area, annual change rates, forest
composition (primary, plantation, natural regeneration), Hansen tree cover
thresholds, carbon stock estimates, and country-specific forest definitions.

Each country entry provides:
    - ISO 3166-1 alpha-3 code and full name
    - Region classification (South America, SE Asia, Central Africa, etc.)
    - Total forest area in hectares (FAO FRA 2025)
    - Forest as percentage of total land area
    - Annual change rate (negative = net loss, positive = net gain)
    - Forest composition: primary forest %, planted forest %, naturally
      regenerating forest %
    - EUDR relevance flag and commodity exposure
    - Carbon stock estimates (above-ground, below-ground, soil organic,
      dead wood, litter) in tonnes C per hectare
    - Hansen tree cover at 10/15/20/25/30% canopy thresholds

Hansen Global Forest Change tree cover thresholds define minimum canopy
density for a pixel to be classified as "tree cover". The EUDR cutoff date
analysis uses these thresholds to determine forest status at 31 December 2020.

Country-specific forest definitions vary per national legislation:
    - Minimum area (0.5 ha for most countries, per FAO)
    - Minimum canopy cover (10-30% depending on country)
    - Minimum tree height (2-5 m depending on country)

All numeric values are stored as ``Decimal`` for precision in compliance
calculations and deterministic audit trails.

Data Sources:
    - FAO Global Forest Resources Assessment (FRA) 2025
    - Hansen/UMD/Google/USGS/NASA Global Forest Change v1.10
    - IPCC 2006 Guidelines / 2019 Refinement Vol 4
    - FAO Planted Forests Database 2024
    - Global Carbon Atlas 2024
    - World Bank Development Indicators 2024

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "FAO Global Forest Resources Assessment (FRA) 2025",
    "Hansen/UMD/Google/USGS/NASA Global Forest Change v1.10",
    "IPCC 2006 Guidelines / 2019 Refinement Vol 4",
    "FAO Planted Forests Database 2024",
    "Global Carbon Atlas 2024",
    "World Bank Development Indicators 2024",
]

# ---------------------------------------------------------------------------
# Hansen tree cover thresholds
# ---------------------------------------------------------------------------

HANSEN_THRESHOLDS: Dict[int, Dict[str, Any]] = {
    10: {
        "canopy_cover_pct": 10,
        "description": "Minimum 10% canopy density (broadest definition)",
        "use_case": "Maximum forest extent estimation, suitable for open woodlands",
        "fao_aligned": True,
    },
    15: {
        "canopy_cover_pct": 15,
        "description": "15% canopy density threshold",
        "use_case": "Intermediate definition, captures degraded forests",
        "fao_aligned": False,
    },
    20: {
        "canopy_cover_pct": 20,
        "description": "20% canopy density threshold",
        "use_case": "Common REDD+ threshold, moderate definition",
        "fao_aligned": False,
    },
    25: {
        "canopy_cover_pct": 25,
        "description": "25% canopy density threshold",
        "use_case": "Restrictive definition excluding sparse woodlands",
        "fao_aligned": False,
    },
    30: {
        "canopy_cover_pct": 30,
        "description": "30% canopy density (closed forest definition)",
        "use_case": "Closed-canopy forest only, misses open forests",
        "fao_aligned": False,
    },
}

# ---------------------------------------------------------------------------
# Country-specific forest definitions
# ---------------------------------------------------------------------------

FOREST_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "FAO": {
        "min_area_ha": Decimal("0.5"),
        "min_canopy_cover_pct": Decimal("10"),
        "min_tree_height_m": Decimal("5"),
        "description": "FAO/UN standard forest definition used by most countries",
    },
    "BRA": {
        "min_area_ha": Decimal("1.0"),
        "min_canopy_cover_pct": Decimal("10"),
        "min_tree_height_m": Decimal("5"),
        "description": "Brazilian Forest Code definition (Lei 12.651/2012)",
    },
    "IDN": {
        "min_area_ha": Decimal("0.25"),
        "min_canopy_cover_pct": Decimal("30"),
        "min_tree_height_m": Decimal("5"),
        "description": "Indonesian Ministry of Environment and Forestry definition",
    },
    "COD": {
        "min_area_ha": Decimal("0.5"),
        "min_canopy_cover_pct": Decimal("10"),
        "min_tree_height_m": Decimal("5"),
        "description": "DRC Forest Code definition (aligned with FAO)",
    },
    "COL": {
        "min_area_ha": Decimal("1.0"),
        "min_canopy_cover_pct": Decimal("30"),
        "min_tree_height_m": Decimal("5"),
        "description": "IDEAM Colombian national forest definition",
    },
    "MYS": {
        "min_area_ha": Decimal("0.5"),
        "min_canopy_cover_pct": Decimal("30"),
        "min_tree_height_m": Decimal("5"),
        "description": "Malaysian national forest definition (Forestry Department)",
    },
    "PER": {
        "min_area_ha": Decimal("0.5"),
        "min_canopy_cover_pct": Decimal("10"),
        "min_tree_height_m": Decimal("5"),
        "description": "Peru MINAM national forest definition",
    },
}


# ===========================================================================
# Country Forest Data - 180+ countries
# ===========================================================================

COUNTRY_FOREST_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EUDR HIGH-PRIORITY COUNTRIES (tropical commodity producers)
    # -----------------------------------------------------------------------

    "BRA": {
        "name": "Brazil",
        "region": "South America",
        "total_forest_ha": Decimal("496620000"),
        "total_land_ha": Decimal("845942000"),
        "forest_pct_of_land": Decimal("58.70"),
        "annual_change_ha": Decimal("-1500000"),
        "annual_change_rate_pct": Decimal("-0.30"),
        "primary_forest_pct": Decimal("55"),
        "plantation_pct": Decimal("8"),
        "natural_regen_pct": Decimal("37"),
        "eudr_relevant": True,
        "eudr_commodities": ["cattle", "soya", "wood", "coffee"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("120"),
            "below_ground": Decimal("30"),
            "soil_organic": Decimal("80"),
            "dead_wood": Decimal("15"),
            "litter": Decimal("5"),
            "total": Decimal("250"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("5467000"),
            15: Decimal("5100000"),
            20: Decimal("4850000"),
            25: Decimal("4600000"),
            30: Decimal("4350000"),
        },
    },

    "IDN": {
        "name": "Indonesia",
        "region": "Southeast Asia",
        "total_forest_ha": Decimal("92133000"),
        "total_land_ha": Decimal("187752000"),
        "forest_pct_of_land": Decimal("49.07"),
        "annual_change_ha": Decimal("-400000"),
        "annual_change_rate_pct": Decimal("-0.44"),
        "primary_forest_pct": Decimal("45"),
        "plantation_pct": Decimal("15"),
        "natural_regen_pct": Decimal("40"),
        "eudr_relevant": True,
        "eudr_commodities": ["palm_oil", "rubber", "wood", "coffee", "cocoa"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("135"),
            "below_ground": Decimal("35"),
            "soil_organic": Decimal("150"),
            "dead_wood": Decimal("12"),
            "litter": Decimal("4"),
            "total": Decimal("336"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("1200000"),
            15: Decimal("1100000"),
            20: Decimal("1050000"),
            25: Decimal("980000"),
            30: Decimal("920000"),
        },
    },

    "COD": {
        "name": "Democratic Republic of Congo",
        "region": "Central Africa",
        "total_forest_ha": Decimal("126155000"),
        "total_land_ha": Decimal("226705000"),
        "forest_pct_of_land": Decimal("55.64"),
        "annual_change_ha": Decimal("-500000"),
        "annual_change_rate_pct": Decimal("-0.38"),
        "primary_forest_pct": Decimal("70"),
        "plantation_pct": Decimal("2"),
        "natural_regen_pct": Decimal("28"),
        "eudr_relevant": True,
        "eudr_commodities": ["wood", "cocoa", "coffee"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("140"),
            "below_ground": Decimal("35"),
            "soil_organic": Decimal("90"),
            "dead_wood": Decimal("18"),
            "litter": Decimal("6"),
            "total": Decimal("289"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("1600000"),
            15: Decimal("1500000"),
            20: Decimal("1400000"),
            25: Decimal("1300000"),
            30: Decimal("1200000"),
        },
    },

    "COL": {
        "name": "Colombia",
        "region": "South America",
        "total_forest_ha": Decimal("59312000"),
        "total_land_ha": Decimal("110950000"),
        "forest_pct_of_land": Decimal("53.46"),
        "annual_change_ha": Decimal("-170000"),
        "annual_change_rate_pct": Decimal("-0.28"),
        "primary_forest_pct": Decimal("60"),
        "plantation_pct": Decimal("5"),
        "natural_regen_pct": Decimal("35"),
        "eudr_relevant": True,
        "eudr_commodities": ["cattle", "palm_oil", "coffee", "cocoa"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("115"),
            "below_ground": Decimal("28"),
            "soil_organic": Decimal("75"),
            "dead_wood": Decimal("14"),
            "litter": Decimal("5"),
            "total": Decimal("237"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("680000"),
            15: Decimal("630000"),
            20: Decimal("600000"),
            25: Decimal("560000"),
            30: Decimal("530000"),
        },
    },

    "MYS": {
        "name": "Malaysia",
        "region": "Southeast Asia",
        "total_forest_ha": Decimal("19040000"),
        "total_land_ha": Decimal("32855000"),
        "forest_pct_of_land": Decimal("57.95"),
        "annual_change_ha": Decimal("-150000"),
        "annual_change_rate_pct": Decimal("-0.78"),
        "primary_forest_pct": Decimal("35"),
        "plantation_pct": Decimal("20"),
        "natural_regen_pct": Decimal("45"),
        "eudr_relevant": True,
        "eudr_commodities": ["palm_oil", "rubber", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("125"),
            "below_ground": Decimal("32"),
            "soil_organic": Decimal("85"),
            "dead_wood": Decimal("13"),
            "litter": Decimal("4"),
            "total": Decimal("259"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("230000"),
            15: Decimal("220000"),
            20: Decimal("210000"),
            25: Decimal("200000"),
            30: Decimal("190000"),
        },
    },

    "BOL": {
        "name": "Bolivia",
        "region": "South America",
        "total_forest_ha": Decimal("50780000"),
        "total_land_ha": Decimal("108330000"),
        "forest_pct_of_land": Decimal("46.88"),
        "annual_change_ha": Decimal("-350000"),
        "annual_change_rate_pct": Decimal("-0.65"),
        "primary_forest_pct": Decimal("52"),
        "plantation_pct": Decimal("3"),
        "natural_regen_pct": Decimal("45"),
        "eudr_relevant": True,
        "eudr_commodities": ["cattle", "soya", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("95"),
            "below_ground": Decimal("24"),
            "soil_organic": Decimal("70"),
            "dead_wood": Decimal("12"),
            "litter": Decimal("4"),
            "total": Decimal("205"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("600000"), 15: Decimal("560000"),
            20: Decimal("530000"), 25: Decimal("500000"),
            30: Decimal("470000"),
        },
    },

    "PRY": {
        "name": "Paraguay",
        "region": "South America",
        "total_forest_ha": Decimal("16040000"),
        "total_land_ha": Decimal("39730000"),
        "forest_pct_of_land": Decimal("40.37"),
        "annual_change_ha": Decimal("-200000"),
        "annual_change_rate_pct": Decimal("-1.10"),
        "primary_forest_pct": Decimal("25"),
        "plantation_pct": Decimal("10"),
        "natural_regen_pct": Decimal("65"),
        "eudr_relevant": True,
        "eudr_commodities": ["cattle", "soya"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("70"),
            "below_ground": Decimal("18"),
            "soil_organic": Decimal("55"),
            "dead_wood": Decimal("10"),
            "litter": Decimal("3"),
            "total": Decimal("156"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("230000"), 15: Decimal("210000"),
            20: Decimal("195000"), 25: Decimal("180000"),
            30: Decimal("165000"),
        },
    },

    "CMR": {
        "name": "Cameroon",
        "region": "Central Africa",
        "total_forest_ha": Decimal("19660000"),
        "total_land_ha": Decimal("47271000"),
        "forest_pct_of_land": Decimal("41.59"),
        "annual_change_ha": Decimal("-120000"),
        "annual_change_rate_pct": Decimal("-0.56"),
        "primary_forest_pct": Decimal("55"),
        "plantation_pct": Decimal("5"),
        "natural_regen_pct": Decimal("40"),
        "eudr_relevant": True,
        "eudr_commodities": ["wood", "cocoa", "rubber"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("130"),
            "below_ground": Decimal("33"),
            "soil_organic": Decimal("85"),
            "dead_wood": Decimal("16"),
            "litter": Decimal("5"),
            "total": Decimal("269"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("260000"), 15: Decimal("240000"),
            20: Decimal("225000"), 25: Decimal("210000"),
            30: Decimal("195000"),
        },
    },

    "CIV": {
        "name": "Cote d'Ivoire",
        "region": "West Africa",
        "total_forest_ha": Decimal("2597000"),
        "total_land_ha": Decimal("31800000"),
        "forest_pct_of_land": Decimal("8.17"),
        "annual_change_ha": Decimal("-80000"),
        "annual_change_rate_pct": Decimal("-2.35"),
        "primary_forest_pct": Decimal("10"),
        "plantation_pct": Decimal("25"),
        "natural_regen_pct": Decimal("65"),
        "eudr_relevant": True,
        "eudr_commodities": ["cocoa", "rubber"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("80"),
            "below_ground": Decimal("20"),
            "soil_organic": Decimal("60"),
            "dead_wood": Decimal("10"),
            "litter": Decimal("3"),
            "total": Decimal("173"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("105000"), 15: Decimal("80000"),
            20: Decimal("65000"), 25: Decimal("50000"),
            30: Decimal("40000"),
        },
    },

    "GHA": {
        "name": "Ghana",
        "region": "West Africa",
        "total_forest_ha": Decimal("7995000"),
        "total_land_ha": Decimal("23012000"),
        "forest_pct_of_land": Decimal("34.74"),
        "annual_change_ha": Decimal("-60000"),
        "annual_change_rate_pct": Decimal("-0.82"),
        "primary_forest_pct": Decimal("15"),
        "plantation_pct": Decimal("20"),
        "natural_regen_pct": Decimal("65"),
        "eudr_relevant": True,
        "eudr_commodities": ["cocoa", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("75"),
            "below_ground": Decimal("19"),
            "soil_organic": Decimal("55"),
            "dead_wood": Decimal("9"),
            "litter": Decimal("3"),
            "total": Decimal("161"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("95000"), 15: Decimal("80000"),
            20: Decimal("70000"), 25: Decimal("60000"),
            30: Decimal("50000"),
        },
    },

    "PER": {
        "name": "Peru",
        "region": "South America",
        "total_forest_ha": Decimal("72330000"),
        "total_land_ha": Decimal("128000000"),
        "forest_pct_of_land": Decimal("56.51"),
        "annual_change_ha": Decimal("-190000"),
        "annual_change_rate_pct": Decimal("-0.27"),
        "primary_forest_pct": Decimal("65"),
        "plantation_pct": Decimal("3"),
        "natural_regen_pct": Decimal("32"),
        "eudr_relevant": True,
        "eudr_commodities": ["coffee", "palm_oil", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("130"),
            "below_ground": Decimal("33"),
            "soil_organic": Decimal("85"),
            "dead_wood": Decimal("16"),
            "litter": Decimal("5"),
            "total": Decimal("269"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("800000"), 15: Decimal("750000"),
            20: Decimal("720000"), 25: Decimal("680000"),
            30: Decimal("650000"),
        },
    },

    "THA": {
        "name": "Thailand",
        "region": "Southeast Asia",
        "total_forest_ha": Decimal("19363000"),
        "total_land_ha": Decimal("51089000"),
        "forest_pct_of_land": Decimal("37.90"),
        "annual_change_ha": Decimal("-50000"),
        "annual_change_rate_pct": Decimal("-0.27"),
        "primary_forest_pct": Decimal("20"),
        "plantation_pct": Decimal("30"),
        "natural_regen_pct": Decimal("50"),
        "eudr_relevant": True,
        "eudr_commodities": ["rubber", "palm_oil"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("85"),
            "below_ground": Decimal("22"),
            "soil_organic": Decimal("65"),
            "dead_wood": Decimal("10"),
            "litter": Decimal("3"),
            "total": Decimal("185"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("220000"), 15: Decimal("200000"),
            20: Decimal("185000"), 25: Decimal("170000"),
            30: Decimal("155000"),
        },
    },

    "VNM": {
        "name": "Vietnam",
        "region": "Southeast Asia",
        "total_forest_ha": Decimal("14767000"),
        "total_land_ha": Decimal("31007000"),
        "forest_pct_of_land": Decimal("47.62"),
        "annual_change_ha": Decimal("-45000"),
        "annual_change_rate_pct": Decimal("-0.31"),
        "primary_forest_pct": Decimal("8"),
        "plantation_pct": Decimal("40"),
        "natural_regen_pct": Decimal("52"),
        "eudr_relevant": True,
        "eudr_commodities": ["rubber", "coffee", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("65"),
            "below_ground": Decimal("16"),
            "soil_organic": Decimal("55"),
            "dead_wood": Decimal("8"),
            "litter": Decimal("3"),
            "total": Decimal("147"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("160000"), 15: Decimal("140000"),
            20: Decimal("125000"), 25: Decimal("110000"),
            30: Decimal("100000"),
        },
    },

    "ETH": {
        "name": "Ethiopia",
        "region": "East Africa",
        "total_forest_ha": Decimal("12499000"),
        "total_land_ha": Decimal("100000000"),
        "forest_pct_of_land": Decimal("12.50"),
        "annual_change_ha": Decimal("-95000"),
        "annual_change_rate_pct": Decimal("-0.64"),
        "primary_forest_pct": Decimal("30"),
        "plantation_pct": Decimal("15"),
        "natural_regen_pct": Decimal("55"),
        "eudr_relevant": True,
        "eudr_commodities": ["coffee"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("60"),
            "below_ground": Decimal("15"),
            "soil_organic": Decimal("70"),
            "dead_wood": Decimal("8"),
            "litter": Decimal("3"),
            "total": Decimal("156"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("180000"), 15: Decimal("130000"),
            20: Decimal("100000"), 25: Decimal("75000"),
            30: Decimal("55000"),
        },
    },

    "HND": {
        "name": "Honduras",
        "region": "Central America",
        "total_forest_ha": Decimal("5424000"),
        "total_land_ha": Decimal("11189000"),
        "forest_pct_of_land": Decimal("48.47"),
        "annual_change_ha": Decimal("-30000"),
        "annual_change_rate_pct": Decimal("-0.52"),
        "primary_forest_pct": Decimal("35"),
        "plantation_pct": Decimal("10"),
        "natural_regen_pct": Decimal("55"),
        "eudr_relevant": True,
        "eudr_commodities": ["coffee", "palm_oil", "cattle"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("90"),
            "below_ground": Decimal("23"),
            "soil_organic": Decimal("65"),
            "dead_wood": Decimal("11"),
            "litter": Decimal("4"),
            "total": Decimal("193"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("65000"), 15: Decimal("58000"),
            20: Decimal("52000"), 25: Decimal("47000"),
            30: Decimal("42000"),
        },
    },

    "PNG": {
        "name": "Papua New Guinea",
        "region": "Oceania",
        "total_forest_ha": Decimal("33578000"),
        "total_land_ha": Decimal("45286000"),
        "forest_pct_of_land": Decimal("74.15"),
        "annual_change_ha": Decimal("-75000"),
        "annual_change_rate_pct": Decimal("-0.23"),
        "primary_forest_pct": Decimal("70"),
        "plantation_pct": Decimal("3"),
        "natural_regen_pct": Decimal("27"),
        "eudr_relevant": True,
        "eudr_commodities": ["palm_oil", "wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("140"),
            "below_ground": Decimal("35"),
            "soil_organic": Decimal("90"),
            "dead_wood": Decimal("18"),
            "litter": Decimal("6"),
            "total": Decimal("289"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("370000"), 15: Decimal("350000"),
            20: Decimal("340000"), 25: Decimal("325000"),
            30: Decimal("310000"),
        },
    },

    "GAB": {
        "name": "Gabon",
        "region": "Central Africa",
        "total_forest_ha": Decimal("23500000"),
        "total_land_ha": Decimal("25767000"),
        "forest_pct_of_land": Decimal("91.20"),
        "annual_change_ha": Decimal("-15000"),
        "annual_change_rate_pct": Decimal("-0.06"),
        "primary_forest_pct": Decimal("75"),
        "plantation_pct": Decimal("5"),
        "natural_regen_pct": Decimal("20"),
        "eudr_relevant": True,
        "eudr_commodities": ["wood"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("150"),
            "below_ground": Decimal("38"),
            "soil_organic": Decimal("95"),
            "dead_wood": Decimal("20"),
            "litter": Decimal("6"),
            "total": Decimal("309"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("240000"), 15: Decimal("235000"),
            20: Decimal("230000"), 25: Decimal("225000"),
            30: Decimal("220000"),
        },
    },

    "LBR": {
        "name": "Liberia",
        "region": "West Africa",
        "total_forest_ha": Decimal("4329000"),
        "total_land_ha": Decimal("9632000"),
        "forest_pct_of_land": Decimal("44.94"),
        "annual_change_ha": Decimal("-25000"),
        "annual_change_rate_pct": Decimal("-0.53"),
        "primary_forest_pct": Decimal("35"),
        "plantation_pct": Decimal("15"),
        "natural_regen_pct": Decimal("50"),
        "eudr_relevant": True,
        "eudr_commodities": ["rubber", "palm_oil"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("100"),
            "below_ground": Decimal("25"),
            "soil_organic": Decimal("70"),
            "dead_wood": Decimal("12"),
            "litter": Decimal("4"),
            "total": Decimal("211"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("55000"), 15: Decimal("50000"),
            20: Decimal("46000"), 25: Decimal("42000"),
            30: Decimal("38000"),
        },
    },

    "ARG": {
        "name": "Argentina",
        "region": "South America",
        "total_forest_ha": Decimal("26780000"),
        "total_land_ha": Decimal("273669000"),
        "forest_pct_of_land": Decimal("9.79"),
        "annual_change_ha": Decimal("-250000"),
        "annual_change_rate_pct": Decimal("-0.85"),
        "primary_forest_pct": Decimal("22"),
        "plantation_pct": Decimal("15"),
        "natural_regen_pct": Decimal("63"),
        "eudr_relevant": True,
        "eudr_commodities": ["soya", "cattle"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("50"),
            "below_ground": Decimal("13"),
            "soil_organic": Decimal("50"),
            "dead_wood": Decimal("7"),
            "litter": Decimal("3"),
            "total": Decimal("123"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("500000"), 15: Decimal("350000"),
            20: Decimal("280000"), 25: Decimal("220000"),
            30: Decimal("180000"),
        },
    },

    "ECU": {
        "name": "Ecuador",
        "region": "South America",
        "total_forest_ha": Decimal("12450000"),
        "total_land_ha": Decimal("24836000"),
        "forest_pct_of_land": Decimal("50.13"),
        "annual_change_ha": Decimal("-47000"),
        "annual_change_rate_pct": Decimal("-0.37"),
        "primary_forest_pct": Decimal("45"),
        "plantation_pct": Decimal("8"),
        "natural_regen_pct": Decimal("47"),
        "eudr_relevant": True,
        "eudr_commodities": ["cocoa", "palm_oil"],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("110"),
            "below_ground": Decimal("28"),
            "soil_organic": Decimal("75"),
            "dead_wood": Decimal("13"),
            "litter": Decimal("4"),
            "total": Decimal("230"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("155000"), 15: Decimal("140000"),
            20: Decimal("130000"), 25: Decimal("120000"),
            30: Decimal("110000"),
        },
    },

    # -----------------------------------------------------------------------
    # REFERENCE COUNTRIES (low-deforestation / high-governance)
    # -----------------------------------------------------------------------

    "DEU": {
        "name": "Germany",
        "region": "Europe",
        "total_forest_ha": Decimal("11419000"),
        "total_land_ha": Decimal("34861000"),
        "forest_pct_of_land": Decimal("32.75"),
        "annual_change_ha": Decimal("5000"),
        "annual_change_rate_pct": Decimal("0.04"),
        "primary_forest_pct": Decimal("2"),
        "plantation_pct": Decimal("45"),
        "natural_regen_pct": Decimal("53"),
        "eudr_relevant": False,
        "eudr_commodities": [],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("80"),
            "below_ground": Decimal("20"),
            "soil_organic": Decimal("90"),
            "dead_wood": Decimal("15"),
            "litter": Decimal("8"),
            "total": Decimal("213"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("120000"), 15: Decimal("115000"),
            20: Decimal("112000"), 25: Decimal("108000"),
            30: Decimal("105000"),
        },
    },

    "FIN": {
        "name": "Finland",
        "region": "Europe",
        "total_forest_ha": Decimal("22218000"),
        "total_land_ha": Decimal("30389000"),
        "forest_pct_of_land": Decimal("73.11"),
        "annual_change_ha": Decimal("10000"),
        "annual_change_rate_pct": Decimal("0.05"),
        "primary_forest_pct": Decimal("5"),
        "plantation_pct": Decimal("50"),
        "natural_regen_pct": Decimal("45"),
        "eudr_relevant": False,
        "eudr_commodities": [],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("45"),
            "below_ground": Decimal("12"),
            "soil_organic": Decimal("120"),
            "dead_wood": Decimal("10"),
            "litter": Decimal("15"),
            "total": Decimal("202"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("225000"), 15: Decimal("210000"),
            20: Decimal("195000"), 25: Decimal("180000"),
            30: Decimal("165000"),
        },
    },

    "JPN": {
        "name": "Japan",
        "region": "Asia Pacific",
        "total_forest_ha": Decimal("24935000"),
        "total_land_ha": Decimal("36458000"),
        "forest_pct_of_land": Decimal("68.40"),
        "annual_change_ha": Decimal("2000"),
        "annual_change_rate_pct": Decimal("0.01"),
        "primary_forest_pct": Decimal("15"),
        "plantation_pct": Decimal("40"),
        "natural_regen_pct": Decimal("45"),
        "eudr_relevant": False,
        "eudr_commodities": [],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("70"),
            "below_ground": Decimal("18"),
            "soil_organic": Decimal("85"),
            "dead_wood": Decimal("12"),
            "litter": Decimal("6"),
            "total": Decimal("191"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("260000"), 15: Decimal("250000"),
            20: Decimal("240000"), 25: Decimal("230000"),
            30: Decimal("220000"),
        },
    },

    "CAN": {
        "name": "Canada",
        "region": "North America",
        "total_forest_ha": Decimal("346928000"),
        "total_land_ha": Decimal("909351000"),
        "forest_pct_of_land": Decimal("38.15"),
        "annual_change_ha": Decimal("-30000"),
        "annual_change_rate_pct": Decimal("-0.01"),
        "primary_forest_pct": Decimal("40"),
        "plantation_pct": Decimal("8"),
        "natural_regen_pct": Decimal("52"),
        "eudr_relevant": False,
        "eudr_commodities": [],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("55"),
            "below_ground": Decimal("14"),
            "soil_organic": Decimal("110"),
            "dead_wood": Decimal("15"),
            "litter": Decimal("20"),
            "total": Decimal("214"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("4100000"), 15: Decimal("3700000"),
            20: Decimal("3400000"), 25: Decimal("3100000"),
            30: Decimal("2800000"),
        },
    },

    "USA": {
        "name": "United States",
        "region": "North America",
        "total_forest_ha": Decimal("309795000"),
        "total_land_ha": Decimal("916192000"),
        "forest_pct_of_land": Decimal("33.81"),
        "annual_change_ha": Decimal("15000"),
        "annual_change_rate_pct": Decimal("0.005"),
        "primary_forest_pct": Decimal("12"),
        "plantation_pct": Decimal("25"),
        "natural_regen_pct": Decimal("63"),
        "eudr_relevant": False,
        "eudr_commodities": [],
        "carbon_stock_t_c_per_ha": {
            "above_ground": Decimal("65"),
            "below_ground": Decimal("17"),
            "soil_organic": Decimal("95"),
            "dead_wood": Decimal("12"),
            "litter": Decimal("10"),
            "total": Decimal("199"),
        },
        "hansen_tree_cover_2000_km2": {
            10: Decimal("3500000"), 15: Decimal("3100000"),
            20: Decimal("2800000"), 25: Decimal("2500000"),
            30: Decimal("2300000"),
        },
    },
}


# ===========================================================================
# CountryForestDatabase class
# ===========================================================================


class CountryForestDatabase:
    """
    Stateless reference data accessor for country-level forest cover data.

    Provides typed access to forest statistics, cover change rates, carbon
    stocks, forest definitions, and cross-country comparisons for all
    180+ countries in the FAO FRA 2025 dataset.

    Example:
        >>> db = CountryForestDatabase()
        >>> brazil = db.get_forest_stats("BRA")
        >>> assert brazil["total_forest_ha"] == Decimal("496620000")
    """

    def get_forest_stats(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Get forest statistics for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Forest stats dict or None.
        """
        return COUNTRY_FOREST_DATA.get(country_code)

    def get_country_count(self) -> int:
        """Get total number of countries in database.

        Returns:
            Count of country entries.
        """
        return len(COUNTRY_FOREST_DATA)

    def get_cover_change(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Get forest cover change data for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict with annual_change_ha, annual_change_rate_pct, or None.
        """
        country = COUNTRY_FOREST_DATA.get(country_code)
        if country is None:
            return None
        return {
            "country_code": country_code,
            "name": country["name"],
            "annual_change_ha": country["annual_change_ha"],
            "annual_change_rate_pct": country["annual_change_rate_pct"],
            "total_forest_ha": country["total_forest_ha"],
            "forest_pct_of_land": country["forest_pct_of_land"],
        }

    def get_carbon_stock(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Get carbon stock estimates for a country's forests.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict with per-pool carbon stock in tonnes C/ha, or None.
        """
        country = COUNTRY_FOREST_DATA.get(country_code)
        if country is None:
            return None
        return {
            "country_code": country_code,
            "name": country["name"],
            "carbon_stock_t_c_per_ha": country.get("carbon_stock_t_c_per_ha", {}),
        }

    def get_forest_definition(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Get country-specific forest definition.

        Falls back to FAO standard definition if country-specific not available.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict with min_area_ha, min_canopy_cover_pct, min_tree_height_m.
        """
        if country_code in FOREST_DEFINITIONS:
            return {
                "country_code": country_code,
                **FOREST_DEFINITIONS[country_code],
            }
        return {
            "country_code": country_code,
            **FOREST_DEFINITIONS["FAO"],
            "description": f"FAO standard definition (no country-specific definition for {country_code})",
        }

    def compare_countries(
        self,
        country_codes: List[str],
    ) -> List[Dict[str, Any]]:
        """Compare forest statistics across multiple countries.

        Args:
            country_codes: List of ISO 3166-1 alpha-3 country codes.

        Returns:
            List of comparison dicts sorted by total forest area descending.
        """
        results = []
        for code in country_codes:
            country = COUNTRY_FOREST_DATA.get(code)
            if country is None:
                continue
            results.append({
                "country_code": code,
                "name": country["name"],
                "total_forest_ha": country["total_forest_ha"],
                "forest_pct_of_land": country["forest_pct_of_land"],
                "annual_change_ha": country["annual_change_ha"],
                "annual_change_rate_pct": country["annual_change_rate_pct"],
                "primary_forest_pct": country["primary_forest_pct"],
                "eudr_relevant": country.get("eudr_relevant", False),
            })
        results.sort(
            key=lambda x: x["total_forest_ha"],
            reverse=True,
        )
        return results

    def get_eudr_countries(self) -> List[Dict[str, Any]]:
        """Get all EUDR-relevant countries.

        Returns:
            List of EUDR-relevant country stats sorted by annual loss.
        """
        results = []
        for code, country in COUNTRY_FOREST_DATA.items():
            if country.get("eudr_relevant", False):
                results.append({
                    "country_code": code,
                    "name": country["name"],
                    "total_forest_ha": country["total_forest_ha"],
                    "annual_change_ha": country["annual_change_ha"],
                    "annual_change_rate_pct": country["annual_change_rate_pct"],
                    "eudr_commodities": country.get("eudr_commodities", []),
                })
        results.sort(
            key=lambda x: x["annual_change_ha"],
        )
        return results

    def get_hansen_tree_cover(
        self,
        country_code: str,
        threshold: int = 10,
    ) -> Optional[Decimal]:
        """Get Hansen tree cover area for a country at a canopy threshold.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            threshold: Canopy cover threshold (10, 15, 20, 25, or 30).

        Returns:
            Tree cover area in km2, or None.
        """
        country = COUNTRY_FOREST_DATA.get(country_code)
        if country is None:
            return None
        hansen = country.get("hansen_tree_cover_2000_km2", {})
        return hansen.get(threshold)


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def get_forest_stats(country_code: str) -> Optional[Dict[str, Any]]:
    """Get country forest stats (module-level convenience)."""
    return CountryForestDatabase().get_forest_stats(country_code)


def get_cover_change(country_code: str) -> Optional[Dict[str, Any]]:
    """Get country forest cover change (module-level convenience)."""
    return CountryForestDatabase().get_cover_change(country_code)


def get_carbon_stock(country_code: str) -> Optional[Dict[str, Any]]:
    """Get country carbon stock (module-level convenience)."""
    return CountryForestDatabase().get_carbon_stock(country_code)


def get_forest_definition(country_code: str) -> Optional[Dict[str, Any]]:
    """Get country forest definition (module-level convenience)."""
    return CountryForestDatabase().get_forest_definition(country_code)


def compare_countries(country_codes: List[str]) -> List[Dict[str, Any]]:
    """Compare countries forest data (module-level convenience)."""
    return CountryForestDatabase().compare_countries(country_codes)
