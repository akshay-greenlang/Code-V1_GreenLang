# -*- coding: utf-8 -*-
"""
Production Statistics - AGENT-EUDR-018 Commodity Risk Analyzer

Global production statistics for all 7 EUDR-regulated commodities including
annual production volumes (tonnes), top-10 producing countries with shares,
year-over-year growth rates, seasonal production patterns (planting and
harvest calendars by region), historical yield data (tonnes/hectare), and
climate sensitivity coefficients.

This data supports:
    - Production forecasting with seasonal adjustment
    - Supply disruption risk assessment via seasonal vulnerability windows
    - Yield estimation for mass balance verification
    - Climate impact modeling on commodity availability
    - Regional production pattern analysis

Seasonal patterns are defined as month-by-month production intensity
coefficients (0.0 = no production, 1.0 = peak production) for major
producing regions. These enable:
    - Detection of off-season declarations (potential fraud indicator)
    - Forecasting supply windows and price pressure periods
    - Identifying climate-sensitive periods per commodity per region

Climate sensitivity coefficients quantify how responsive each commodity
is to temperature and rainfall deviations from baseline, critical for
assessing production risk under changing climate conditions.

Data Sources:
    - FAO FAOSTAT Production Statistics 2024
    - USDA Foreign Agricultural Service (FAS) Production Estimates 2024
    - International Cocoa Organization (ICCO) Quarterly Bulletin 2024
    - International Coffee Organization (ICO) Market Reports 2024
    - Malaysian Palm Oil Board (MPOB) Statistics 2024
    - International Rubber Study Group (IRSG) Outlook 2024
    - ITTO Annual Review and Assessment 2024

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "FAO FAOSTAT Production Statistics 2024",
    "USDA Foreign Agricultural Service (FAS) Production Estimates 2024",
    "International Cocoa Organization (ICCO) Quarterly Bulletin 2024",
    "International Coffee Organization (ICO) Market Reports 2024",
    "Malaysian Palm Oil Board (MPOB) Statistics 2024",
    "International Rubber Study Group (IRSG) Outlook 2024",
    "ITTO Annual Review and Assessment of the World Timber Situation 2024",
]

# ===========================================================================
# Production Statistics - Per-commodity global data
# ===========================================================================

PRODUCTION_STATISTICS: Dict[str, Dict[str, Any]] = {

    "cattle": {
        "commodity_type": "cattle",
        "global_production_tonnes": 72_500_000,
        "production_year": 2024,
        "unit": "tonnes_carcass_weight",
        "yoy_growth_rate": 0.012,
        "five_year_cagr": 0.008,
        "top_producers": [
            {"country": "USA", "volume_tonnes": 12_300_000, "share": 0.170, "rank": 1},
            {"country": "BRA", "volume_tonnes": 10_200_000, "share": 0.141, "rank": 2},
            {"country": "CHN", "volume_tonnes": 7_200_000, "share": 0.099, "rank": 3},
            {"country": "ARG", "volume_tonnes": 3_100_000, "share": 0.043, "rank": 4},
            {"country": "AUS", "volume_tonnes": 2_100_000, "share": 0.029, "rank": 5},
            {"country": "IND", "volume_tonnes": 4_300_000, "share": 0.059, "rank": 6},
            {"country": "MEX", "volume_tonnes": 2_200_000, "share": 0.030, "rank": 7},
            {"country": "PAK", "volume_tonnes": 2_100_000, "share": 0.029, "rank": 8},
            {"country": "RUS", "volume_tonnes": 1_600_000, "share": 0.022, "rank": 9},
            {"country": "COL", "volume_tonnes": 1_000_000, "share": 0.014, "rank": 10},
        ],
        "historical_yields": {
            "BRA": {"yield_per_hectare": 1.8, "unit": "head_per_hectare", "year": 2024},
            "USA": {"yield_per_hectare": 3.2, "unit": "head_per_hectare", "year": 2024},
            "ARG": {"yield_per_hectare": 1.5, "unit": "head_per_hectare", "year": 2024},
            "AUS": {"yield_per_hectare": 0.4, "unit": "head_per_hectare", "year": 2024},
        },
    },

    "cocoa": {
        "commodity_type": "cocoa",
        "global_production_tonnes": 5_800_000,
        "production_year": 2024,
        "unit": "tonnes_dry_beans",
        "yoy_growth_rate": -0.035,
        "five_year_cagr": 0.005,
        "top_producers": [
            {"country": "CIV", "volume_tonnes": 2_200_000, "share": 0.379, "rank": 1},
            {"country": "GHA", "volume_tonnes": 1_050_000, "share": 0.181, "rank": 2},
            {"country": "ECU", "volume_tonnes": 380_000, "share": 0.066, "rank": 3},
            {"country": "CMR", "volume_tonnes": 290_000, "share": 0.050, "rank": 4},
            {"country": "NGA", "volume_tonnes": 270_000, "share": 0.047, "rank": 5},
            {"country": "IDN", "volume_tonnes": 200_000, "share": 0.034, "rank": 6},
            {"country": "BRA", "volume_tonnes": 175_000, "share": 0.030, "rank": 7},
            {"country": "PER", "volume_tonnes": 160_000, "share": 0.028, "rank": 8},
            {"country": "DOM", "volume_tonnes": 85_000, "share": 0.015, "rank": 9},
            {"country": "COL", "volume_tonnes": 65_000, "share": 0.011, "rank": 10},
        ],
        "historical_yields": {
            "CIV": {"yield_per_hectare": 0.55, "unit": "tonnes_per_hectare", "year": 2024},
            "GHA": {"yield_per_hectare": 0.45, "unit": "tonnes_per_hectare", "year": 2024},
            "ECU": {"yield_per_hectare": 0.50, "unit": "tonnes_per_hectare", "year": 2024},
            "IDN": {"yield_per_hectare": 0.40, "unit": "tonnes_per_hectare", "year": 2024},
        },
    },

    "coffee": {
        "commodity_type": "coffee",
        "global_production_tonnes": 10_800_000,
        "production_year": 2024,
        "unit": "tonnes_green_bean_equivalent",
        "yoy_growth_rate": 0.015,
        "five_year_cagr": 0.018,
        "top_producers": [
            {"country": "BRA", "volume_tonnes": 3_900_000, "share": 0.361, "rank": 1},
            {"country": "VNM", "volume_tonnes": 1_850_000, "share": 0.171, "rank": 2},
            {"country": "COL", "volume_tonnes": 870_000, "share": 0.081, "rank": 3},
            {"country": "IDN", "volume_tonnes": 750_000, "share": 0.069, "rank": 4},
            {"country": "ETH", "volume_tonnes": 530_000, "share": 0.049, "rank": 5},
            {"country": "HND", "volume_tonnes": 420_000, "share": 0.039, "rank": 6},
            {"country": "IND", "volume_tonnes": 380_000, "share": 0.035, "rank": 7},
            {"country": "UGA", "volume_tonnes": 340_000, "share": 0.031, "rank": 8},
            {"country": "MEX", "volume_tonnes": 280_000, "share": 0.026, "rank": 9},
            {"country": "GTM", "volume_tonnes": 250_000, "share": 0.023, "rank": 10},
        ],
        "historical_yields": {
            "BRA": {"yield_per_hectare": 1.65, "unit": "tonnes_per_hectare", "year": 2024},
            "VNM": {"yield_per_hectare": 2.80, "unit": "tonnes_per_hectare", "year": 2024},
            "COL": {"yield_per_hectare": 1.10, "unit": "tonnes_per_hectare", "year": 2024},
            "ETH": {"yield_per_hectare": 0.75, "unit": "tonnes_per_hectare", "year": 2024},
        },
    },

    "oil_palm": {
        "commodity_type": "oil_palm",
        "global_production_tonnes": 78_000_000,
        "production_year": 2024,
        "unit": "tonnes_crude_palm_oil",
        "yoy_growth_rate": 0.025,
        "five_year_cagr": 0.030,
        "top_producers": [
            {"country": "IDN", "volume_tonnes": 49_500_000, "share": 0.635, "rank": 1},
            {"country": "MYS", "volume_tonnes": 18_500_000, "share": 0.237, "rank": 2},
            {"country": "THA", "volume_tonnes": 3_200_000, "share": 0.041, "rank": 3},
            {"country": "COL", "volume_tonnes": 1_800_000, "share": 0.023, "rank": 4},
            {"country": "NGA", "volume_tonnes": 1_400_000, "share": 0.018, "rank": 5},
            {"country": "GTM", "volume_tonnes": 800_000, "share": 0.010, "rank": 6},
            {"country": "HND", "volume_tonnes": 650_000, "share": 0.008, "rank": 7},
            {"country": "PNG", "volume_tonnes": 600_000, "share": 0.008, "rank": 8},
            {"country": "ECU", "volume_tonnes": 500_000, "share": 0.006, "rank": 9},
            {"country": "GHA", "volume_tonnes": 400_000, "share": 0.005, "rank": 10},
        ],
        "historical_yields": {
            "IDN": {"yield_per_hectare": 3.50, "unit": "tonnes_cpo_per_hectare", "year": 2024},
            "MYS": {"yield_per_hectare": 3.80, "unit": "tonnes_cpo_per_hectare", "year": 2024},
            "THA": {"yield_per_hectare": 3.20, "unit": "tonnes_cpo_per_hectare", "year": 2024},
            "COL": {"yield_per_hectare": 3.60, "unit": "tonnes_cpo_per_hectare", "year": 2024},
        },
    },

    "rubber": {
        "commodity_type": "rubber",
        "global_production_tonnes": 14_500_000,
        "production_year": 2024,
        "unit": "tonnes_dry_rubber",
        "yoy_growth_rate": 0.008,
        "five_year_cagr": 0.010,
        "top_producers": [
            {"country": "THA", "volume_tonnes": 4_700_000, "share": 0.324, "rank": 1},
            {"country": "IDN", "volume_tonnes": 3_100_000, "share": 0.214, "rank": 2},
            {"country": "VNM", "volume_tonnes": 1_300_000, "share": 0.090, "rank": 3},
            {"country": "CHN", "volume_tonnes": 850_000, "share": 0.059, "rank": 4},
            {"country": "IND", "volume_tonnes": 780_000, "share": 0.054, "rank": 5},
            {"country": "MYS", "volume_tonnes": 500_000, "share": 0.034, "rank": 6},
            {"country": "CIV", "volume_tonnes": 450_000, "share": 0.031, "rank": 7},
            {"country": "GTM", "volume_tonnes": 250_000, "share": 0.017, "rank": 8},
            {"country": "MMR", "volume_tonnes": 200_000, "share": 0.014, "rank": 9},
            {"country": "CMR", "volume_tonnes": 150_000, "share": 0.010, "rank": 10},
        ],
        "historical_yields": {
            "THA": {"yield_per_hectare": 1.65, "unit": "tonnes_per_hectare", "year": 2024},
            "IDN": {"yield_per_hectare": 1.10, "unit": "tonnes_per_hectare", "year": 2024},
            "VNM": {"yield_per_hectare": 1.70, "unit": "tonnes_per_hectare", "year": 2024},
            "MYS": {"yield_per_hectare": 1.20, "unit": "tonnes_per_hectare", "year": 2024},
        },
    },

    "soya": {
        "commodity_type": "soya",
        "global_production_tonnes": 395_000_000,
        "production_year": 2024,
        "unit": "tonnes_soybeans",
        "yoy_growth_rate": 0.028,
        "five_year_cagr": 0.035,
        "top_producers": [
            {"country": "BRA", "volume_tonnes": 160_000_000, "share": 0.405, "rank": 1},
            {"country": "USA", "volume_tonnes": 121_000_000, "share": 0.306, "rank": 2},
            {"country": "ARG", "volume_tonnes": 50_000_000, "share": 0.127, "rank": 3},
            {"country": "CHN", "volume_tonnes": 20_500_000, "share": 0.052, "rank": 4},
            {"country": "IND", "volume_tonnes": 12_000_000, "share": 0.030, "rank": 5},
            {"country": "PRY", "volume_tonnes": 10_500_000, "share": 0.027, "rank": 6},
            {"country": "CAN", "volume_tonnes": 7_500_000, "share": 0.019, "rank": 7},
            {"country": "BOL", "volume_tonnes": 3_200_000, "share": 0.008, "rank": 8},
            {"country": "URY", "volume_tonnes": 3_000_000, "share": 0.008, "rank": 9},
            {"country": "UKR", "volume_tonnes": 2_800_000, "share": 0.007, "rank": 10},
        ],
        "historical_yields": {
            "BRA": {"yield_per_hectare": 3.45, "unit": "tonnes_per_hectare", "year": 2024},
            "USA": {"yield_per_hectare": 3.35, "unit": "tonnes_per_hectare", "year": 2024},
            "ARG": {"yield_per_hectare": 2.90, "unit": "tonnes_per_hectare", "year": 2024},
            "PRY": {"yield_per_hectare": 2.60, "unit": "tonnes_per_hectare", "year": 2024},
        },
    },

    "wood": {
        "commodity_type": "wood",
        "global_production_tonnes": 4_000_000_000,
        "production_year": 2024,
        "unit": "cubic_metres_roundwood_equivalent",
        "yoy_growth_rate": 0.005,
        "five_year_cagr": 0.003,
        "top_producers": [
            {"country": "USA", "volume_tonnes": 420_000_000, "share": 0.105, "rank": 1},
            {"country": "CHN", "volume_tonnes": 380_000_000, "share": 0.095, "rank": 2},
            {"country": "BRA", "volume_tonnes": 260_000_000, "share": 0.065, "rank": 3},
            {"country": "RUS", "volume_tonnes": 220_000_000, "share": 0.055, "rank": 4},
            {"country": "CAN", "volume_tonnes": 150_000_000, "share": 0.038, "rank": 5},
            {"country": "IDN", "volume_tonnes": 120_000_000, "share": 0.030, "rank": 6},
            {"country": "IND", "volume_tonnes": 360_000_000, "share": 0.090, "rank": 7},
            {"country": "SWE", "volume_tonnes": 75_000_000, "share": 0.019, "rank": 8},
            {"country": "FIN", "volume_tonnes": 62_000_000, "share": 0.016, "rank": 9},
            {"country": "DEU", "volume_tonnes": 55_000_000, "share": 0.014, "rank": 10},
        ],
        "historical_yields": {
            "BRA": {"yield_per_hectare": 35.0, "unit": "cubic_metres_per_hectare_per_year", "year": 2024},
            "SWE": {"yield_per_hectare": 5.5, "unit": "cubic_metres_per_hectare_per_year", "year": 2024},
            "FIN": {"yield_per_hectare": 4.8, "unit": "cubic_metres_per_hectare_per_year", "year": 2024},
            "IDN": {"yield_per_hectare": 20.0, "unit": "cubic_metres_per_hectare_per_year", "year": 2024},
        },
    },
}

# ===========================================================================
# Seasonal Patterns - Month-by-month production intensity per region
# ===========================================================================
# Values: 0.0 (no production) to 1.0 (peak production)
# Months: 1=Jan, 2=Feb, ..., 12=Dec

SEASONAL_PATTERNS: Dict[str, Dict[str, Dict[str, Any]]] = {

    "cattle": {
        "BRA": {
            "description": "Year-round, peak processing May-Aug (dry season)",
            "months": {1: 0.75, 2: 0.70, 3: 0.75, 4: 0.80, 5: 0.95, 6: 1.00, 7: 1.00, 8: 0.95, 9: 0.85, 10: 0.80, 11: 0.75, 12: 0.70},
            "planting_months": [],
            "harvest_months": [5, 6, 7, 8],
        },
        "USA": {
            "description": "Year-round, peak processing Oct-Dec",
            "months": {1: 0.80, 2: 0.75, 3: 0.80, 4: 0.85, 5: 0.85, 6: 0.85, 7: 0.80, 8: 0.80, 9: 0.85, 10: 0.95, 11: 1.00, 12: 0.95},
            "planting_months": [],
            "harvest_months": [10, 11, 12],
        },
        "ARG": {
            "description": "Year-round, peak processing Mar-Jun",
            "months": {1: 0.70, 2: 0.75, 3: 0.90, 4: 0.95, 5: 1.00, 6: 0.95, 7: 0.85, 8: 0.80, 9: 0.75, 10: 0.75, 11: 0.70, 12: 0.65},
            "planting_months": [],
            "harvest_months": [3, 4, 5, 6],
        },
    },

    "cocoa": {
        "CIV": {
            "description": "Main crop Oct-Mar, mid-crop May-Aug",
            "months": {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.30, 5: 0.45, 6: 0.55, 7: 0.60, 8: 0.50, 9: 0.35, 10: 0.90, 11: 1.00, 12: 0.95},
            "planting_months": [],
            "harvest_months": [10, 11, 12, 1, 2, 3, 5, 6, 7, 8],
        },
        "GHA": {
            "description": "Main crop Oct-Mar, mid-crop May-Aug",
            "months": {1: 0.80, 2: 0.65, 3: 0.50, 4: 0.25, 5: 0.40, 6: 0.50, 7: 0.55, 8: 0.45, 9: 0.30, 10: 0.85, 11: 1.00, 12: 0.90},
            "planting_months": [],
            "harvest_months": [10, 11, 12, 1, 2, 3, 5, 6, 7, 8],
        },
        "ECU": {
            "description": "Year-round with peak Mar-Jun",
            "months": {1: 0.60, 2: 0.70, 3: 0.90, 4: 1.00, 5: 0.95, 6: 0.85, 7: 0.70, 8: 0.55, 9: 0.50, 10: 0.50, 11: 0.55, 12: 0.55},
            "planting_months": [],
            "harvest_months": [3, 4, 5, 6],
        },
    },

    "coffee": {
        "BRA": {
            "description": "Main harvest May-Sep (arabica), Jun-Oct (robusta)",
            "months": {1: 0.15, 2: 0.10, 3: 0.10, 4: 0.20, 5: 0.65, 6: 0.90, 7: 1.00, 8: 0.95, 9: 0.70, 10: 0.30, 11: 0.15, 12: 0.10},
            "planting_months": [10, 11, 12],
            "harvest_months": [5, 6, 7, 8, 9],
        },
        "COL": {
            "description": "Main harvest Oct-Dec, mitaca Apr-Jun",
            "months": {1: 0.25, 2: 0.20, 3: 0.25, 4: 0.55, 5: 0.65, 6: 0.50, 7: 0.30, 8: 0.25, 9: 0.35, 10: 0.80, 11: 1.00, 12: 0.90},
            "planting_months": [1, 2, 3],
            "harvest_months": [10, 11, 12, 4, 5, 6],
        },
        "VNM": {
            "description": "Main harvest Nov-Feb (robusta)",
            "months": {1: 0.80, 2: 0.50, 3: 0.20, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.10, 9: 0.15, 10: 0.30, 11: 0.90, 12: 1.00},
            "planting_months": [5, 6, 7],
            "harvest_months": [11, 12, 1, 2],
        },
        "ETH": {
            "description": "Main harvest Oct-Jan (wild and cultivated)",
            "months": {1: 0.60, 2: 0.30, 3: 0.15, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.15, 9: 0.25, 10: 0.70, 11: 0.95, 12: 1.00},
            "planting_months": [4, 5, 6],
            "harvest_months": [10, 11, 12, 1],
        },
    },

    "oil_palm": {
        "IDN": {
            "description": "Year-round with peak Mar-Apr, Oct-Nov",
            "months": {1: 0.75, 2: 0.80, 3: 0.95, 4: 1.00, 5: 0.90, 6: 0.80, 7: 0.75, 8: 0.75, 9: 0.85, 10: 0.95, 11: 0.95, 12: 0.80},
            "planting_months": [],
            "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
        "MYS": {
            "description": "Year-round with peak Aug-Nov",
            "months": {1: 0.65, 2: 0.60, 3: 0.70, 4: 0.75, 5: 0.80, 6: 0.80, 7: 0.85, 8: 0.95, 9: 1.00, 10: 0.95, 11: 0.90, 12: 0.70},
            "planting_months": [],
            "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
        "COL": {
            "description": "Year-round with peak Feb-May",
            "months": {1: 0.70, 2: 0.85, 3: 0.95, 4: 1.00, 5: 0.90, 6: 0.80, 7: 0.75, 8: 0.70, 9: 0.75, 10: 0.80, 11: 0.75, 12: 0.65},
            "planting_months": [],
            "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    },

    "rubber": {
        "THA": {
            "description": "Peak Mar-May, low Dec-Feb (wintering)",
            "months": {1: 0.30, 2: 0.40, 3: 0.85, 4: 0.95, 5: 1.00, 6: 0.90, 7: 0.85, 8: 0.85, 9: 0.80, 10: 0.75, 11: 0.55, 12: 0.25},
            "planting_months": [],
            "harvest_months": [3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        "IDN": {
            "description": "Year-round, reduced Feb-Mar (wintering)",
            "months": {1: 0.55, 2: 0.40, 3: 0.45, 4: 0.70, 5: 0.85, 6: 0.90, 7: 0.95, 8: 1.00, 9: 0.95, 10: 0.90, 11: 0.80, 12: 0.65},
            "planting_months": [],
            "harvest_months": [4, 5, 6, 7, 8, 9, 10, 11],
        },
        "VNM": {
            "description": "Peak May-Nov, wintering Dec-Mar",
            "months": {1: 0.20, 2: 0.15, 3: 0.30, 4: 0.50, 5: 0.80, 6: 0.90, 7: 1.00, 8: 0.95, 9: 0.90, 10: 0.85, 11: 0.60, 12: 0.25},
            "planting_months": [],
            "harvest_months": [5, 6, 7, 8, 9, 10, 11],
        },
    },

    "soya": {
        "BRA": {
            "description": "Plant Oct-Dec, harvest Feb-May (main), safrinha Jan-Mar harvest Jun-Aug",
            "months": {1: 0.15, 2: 0.55, 3: 0.90, 4: 1.00, 5: 0.70, 6: 0.25, 7: 0.30, 8: 0.20, 9: 0.10, 10: 0.10, 11: 0.10, 12: 0.10},
            "planting_months": [10, 11, 12],
            "harvest_months": [2, 3, 4, 5],
        },
        "USA": {
            "description": "Plant May-Jun, harvest Sep-Nov",
            "months": {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.15, 9: 0.65, 10: 1.00, 11: 0.70, 12: 0.10},
            "planting_months": [5, 6],
            "harvest_months": [9, 10, 11],
        },
        "ARG": {
            "description": "Plant Nov-Dec, harvest Mar-May",
            "months": {1: 0.10, 2: 0.15, 3: 0.70, 4: 1.00, 5: 0.60, 6: 0.10, 7: 0.05, 8: 0.05, 9: 0.05, 10: 0.05, 11: 0.08, 12: 0.10},
            "planting_months": [11, 12],
            "harvest_months": [3, 4, 5],
        },
    },

    "wood": {
        "BRA": {
            "description": "Year-round, reduced during wet season (Dec-Mar)",
            "months": {1: 0.65, 2: 0.60, 3: 0.70, 4: 0.85, 5: 0.90, 6: 0.95, 7: 1.00, 8: 1.00, 9: 0.95, 10: 0.90, 11: 0.80, 12: 0.65},
            "planting_months": [],
            "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
        "IDN": {
            "description": "Year-round, reduced during monsoon (Nov-Mar)",
            "months": {1: 0.55, 2: 0.50, 3: 0.55, 4: 0.70, 5: 0.85, 6: 0.95, 7: 1.00, 8: 1.00, 9: 0.95, 10: 0.85, 11: 0.65, 12: 0.50},
            "planting_months": [],
            "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
        "SWE": {
            "description": "Peak logging during winter (frozen ground), reduced spring (thaw)",
            "months": {1: 1.00, 2: 0.95, 3: 0.80, 4: 0.40, 5: 0.50, 6: 0.65, 7: 0.70, 8: 0.75, 9: 0.80, 10: 0.85, 11: 0.90, 12: 0.95},
            "planting_months": [4, 5],
            "harvest_months": [1, 2, 3, 9, 10, 11, 12],
        },
        "CAN": {
            "description": "Peak logging winter, reduced spring breakup",
            "months": {1: 0.95, 2: 1.00, 3: 0.85, 4: 0.35, 5: 0.40, 6: 0.60, 7: 0.70, 8: 0.75, 9: 0.80, 10: 0.85, 11: 0.90, 12: 0.95},
            "planting_months": [5, 6],
            "harvest_months": [1, 2, 3, 9, 10, 11, 12],
        },
    },
}

# ===========================================================================
# Climate Sensitivity - Temperature and rainfall sensitivity coefficients
# ===========================================================================
# Higher values = more sensitive to deviations from normal conditions

CLIMATE_SENSITIVITY: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "temperature_sensitivity": 0.35,
        "rainfall_sensitivity": 0.45,
        "drought_vulnerability": 0.60,
        "flood_vulnerability": 0.30,
        "description": "Moderate sensitivity; drought affects pasture quality and feed availability",
        "critical_temp_range_celsius": {"min": -5, "max": 38},
        "optimal_temp_range_celsius": {"min": 10, "max": 28},
    },
    "cocoa": {
        "temperature_sensitivity": 0.70,
        "rainfall_sensitivity": 0.80,
        "drought_vulnerability": 0.85,
        "flood_vulnerability": 0.50,
        "description": "High sensitivity; requires specific temperature and humidity ranges, highly vulnerable to drought",
        "critical_temp_range_celsius": {"min": 15, "max": 35},
        "optimal_temp_range_celsius": {"min": 21, "max": 30},
    },
    "coffee": {
        "temperature_sensitivity": 0.75,
        "rainfall_sensitivity": 0.70,
        "drought_vulnerability": 0.75,
        "flood_vulnerability": 0.45,
        "description": "High temperature sensitivity; frost damage in arabica, drought reduces cherry quality",
        "critical_temp_range_celsius": {"min": 5, "max": 35},
        "optimal_temp_range_celsius": {"min": 15, "max": 24},
    },
    "oil_palm": {
        "temperature_sensitivity": 0.50,
        "rainfall_sensitivity": 0.65,
        "drought_vulnerability": 0.70,
        "flood_vulnerability": 0.35,
        "description": "Moderate sensitivity; consistent rainfall critical, El Nino events reduce yields 6-18 months later",
        "critical_temp_range_celsius": {"min": 18, "max": 38},
        "optimal_temp_range_celsius": {"min": 24, "max": 32},
    },
    "rubber": {
        "temperature_sensitivity": 0.55,
        "rainfall_sensitivity": 0.60,
        "drought_vulnerability": 0.55,
        "flood_vulnerability": 0.40,
        "description": "Moderate sensitivity; wintering period natural, prolonged drought reduces latex flow",
        "critical_temp_range_celsius": {"min": 15, "max": 38},
        "optimal_temp_range_celsius": {"min": 22, "max": 32},
    },
    "soya": {
        "temperature_sensitivity": 0.60,
        "rainfall_sensitivity": 0.75,
        "drought_vulnerability": 0.80,
        "flood_vulnerability": 0.55,
        "description": "High rainfall sensitivity during pod-fill stage; drought during flowering severely reduces yield",
        "critical_temp_range_celsius": {"min": 5, "max": 40},
        "optimal_temp_range_celsius": {"min": 20, "max": 30},
    },
    "wood": {
        "temperature_sensitivity": 0.30,
        "rainfall_sensitivity": 0.35,
        "drought_vulnerability": 0.40,
        "flood_vulnerability": 0.25,
        "description": "Lower sensitivity for mature timber; fire risk increases significantly during drought periods",
        "critical_temp_range_celsius": {"min": -40, "max": 45},
        "optimal_temp_range_celsius": {"min": 10, "max": 30},
    },
}


# ===========================================================================
# ProductionStatistics class
# ===========================================================================


class ProductionStatistics:
    """
    Stateless accessor for global production statistics.

    Provides methods to query production volumes, seasonal patterns,
    historical yields, and climate sensitivity data for all 7 EUDR
    commodities.

    Example:
        >>> stats = ProductionStatistics()
        >>> cocoa = stats.get_production_stats("cocoa")
        >>> assert cocoa["global_production_tonnes"] > 0
        >>> pattern = stats.get_seasonal_pattern("coffee", "BRA")
        >>> assert 7 in pattern["harvest_months"]
    """

    def get_production_stats(
        self, commodity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get production statistics for a commodity.

        Args:
            commodity_type: One of the 7 EUDR commodity types.

        Returns:
            Production statistics dict or None if not found.
        """
        return PRODUCTION_STATISTICS.get(commodity_type)

    def get_seasonal_pattern(
        self, commodity_type: str, country: str
    ) -> Optional[Dict[str, Any]]:
        """Get seasonal production pattern for a commodity and country.

        Args:
            commodity_type: EUDR commodity type.
            country: ISO 3166-1 alpha-3 country code.

        Returns:
            Seasonal pattern dict or None if not found.
        """
        commodity_patterns = SEASONAL_PATTERNS.get(commodity_type)
        if commodity_patterns is None:
            return None
        return commodity_patterns.get(country)

    def get_yield_data(
        self, commodity_type: str, country: str
    ) -> Optional[Dict[str, Any]]:
        """Get historical yield data for a commodity and country.

        Args:
            commodity_type: EUDR commodity type.
            country: ISO 3166-1 alpha-3 country code.

        Returns:
            Yield data dict or None if not found.
        """
        stats = PRODUCTION_STATISTICS.get(commodity_type)
        if stats is None:
            return None
        return stats.get("historical_yields", {}).get(country)

    def get_top_producers(
        self, commodity_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top producing countries for a commodity.

        Args:
            commodity_type: EUDR commodity type.
            limit: Maximum number of producers to return.

        Returns:
            List of producer dicts sorted by rank (empty if not found).
        """
        stats = PRODUCTION_STATISTICS.get(commodity_type)
        if stats is None:
            return []
        return stats.get("top_producers", [])[:limit]


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_production_stats(commodity_type: str) -> Optional[Dict[str, Any]]:
    """Get production statistics for a commodity.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        Production statistics dict or None if not found.
    """
    return PRODUCTION_STATISTICS.get(commodity_type)


def get_seasonal_pattern(
    commodity_type: str, country: str
) -> Optional[Dict[str, Any]]:
    """Get seasonal production pattern for a commodity and country.

    Args:
        commodity_type: EUDR commodity type.
        country: ISO 3166-1 alpha-3 country code.

    Returns:
        Seasonal pattern dict or None if not found.
    """
    commodity_patterns = SEASONAL_PATTERNS.get(commodity_type)
    if commodity_patterns is None:
        return None
    return commodity_patterns.get(country)


def get_yield_data(
    commodity_type: str, country: str
) -> Optional[Dict[str, Any]]:
    """Get historical yield data for a commodity and country.

    Args:
        commodity_type: EUDR commodity type.
        country: ISO 3166-1 alpha-3 country code.

    Returns:
        Yield data dict or None if not found.
    """
    stats = PRODUCTION_STATISTICS.get(commodity_type)
    if stats is None:
        return None
    return stats.get("historical_yields", {}).get(country)


def get_top_producers(
    commodity_type: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Get top producing countries for a commodity.

    Args:
        commodity_type: EUDR commodity type.
        limit: Maximum number of producers to return.

    Returns:
        List of producer dicts sorted by rank (empty if not found).
    """
    stats = PRODUCTION_STATISTICS.get(commodity_type)
    if stats is None:
        return []
    return stats.get("top_producers", [])[:limit]


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "PRODUCTION_STATISTICS",
    "SEASONAL_PATTERNS",
    "CLIMATE_SENSITIVITY",
    "ProductionStatistics",
    "get_production_stats",
    "get_seasonal_pattern",
    "get_yield_data",
    "get_top_producers",
]
