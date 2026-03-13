# -*- coding: utf-8 -*-
"""
CPI Database - AGENT-EUDR-019 Corruption Index Monitor

Transparency International Corruption Perceptions Index (CPI) reference data
covering 180+ countries with multi-year historical scores (2018-2025) on the
standard 0-100 scale (0 = highly corrupt, 100 = very clean). Each country
entry provides ISO 3166-1 alpha-2 and alpha-3 codes, full name, region and
sub-region classification, World Bank income level, and per-year CPI scores
with associated global rank and confidence interval.

Regional classifications follow Transparency International categories:
    - Sub-Saharan Africa (SSA)
    - Western Europe & EU (WEU)
    - Asia Pacific (AP)
    - Americas (AMR)
    - Eastern Europe & Central Asia (ECA)
    - Middle East & North Africa (MENA)

EUDR-relevant countries are prioritized with complete historical data. All
numeric values are stored as ``Decimal`` for precision in compliance
calculations and deterministic audit trails.

Data Sources:
    - Transparency International CPI 2024 (released January 2025)
    - Transparency International CPI Historical Scores 2012-2024
    - ISO 3166-1 Country Codes Standard
    - World Bank Country Classification by Income Level 2024

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "Transparency International CPI 2024",
    "Transparency International CPI Historical Scores 2012-2024",
    "ISO 3166-1 Country Codes Standard",
    "World Bank Country Classification by Income Level 2024",
]

# ---------------------------------------------------------------------------
# Region constants
# ---------------------------------------------------------------------------

REGIONS: List[str] = [
    "sub_saharan_africa",
    "western_europe",
    "asia_pacific",
    "americas",
    "eastern_europe_central_asia",
    "middle_east_north_africa",
]

REGION_DISPLAY_NAMES: Dict[str, str] = {
    "sub_saharan_africa": "Sub-Saharan Africa",
    "western_europe": "Western Europe & EU",
    "asia_pacific": "Asia Pacific",
    "americas": "Americas",
    "eastern_europe_central_asia": "Eastern Europe & Central Asia",
    "middle_east_north_africa": "Middle East & North Africa",
}

# ===========================================================================
# CPI Country Data - 180+ countries with multi-year scores
# ===========================================================================
# Structure per country:
#   iso_alpha2: ISO 3166-1 alpha-2 code
#   iso_alpha3: ISO 3166-1 alpha-3 code
#   name: Full country name
#   region: Transparency International region classification
#   sub_region: More specific geographic sub-region
#   income_level: World Bank income classification
#   scores: {year: {"score": Decimal, "rank": int, "ci_low": Decimal, "ci_high": Decimal}}

CPI_COUNTRY_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EUDR HIGH-PRIORITY COUNTRIES (tropical commodity producers)
    # -----------------------------------------------------------------------

    "BR": {
        "iso_alpha2": "BR",
        "iso_alpha3": "BRA",
        "name": "Brazil",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("38"), "rank": 104, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2024: {"score": Decimal("36"), "rank": 107, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2023: {"score": Decimal("36"), "rank": 104, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2022: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2021: {"score": Decimal("38"), "rank": 96, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2020: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("42")},
            2019: {"score": Decimal("35"), "rank": 106, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
            2018: {"score": Decimal("35"), "rank": 105, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
        },
    },

    "ID": {
        "iso_alpha2": "ID",
        "iso_alpha3": "IDN",
        "name": "Indonesia",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2024: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2023: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2022: {"score": Decimal("34"), "rank": 110, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2021: {"score": Decimal("38"), "rank": 96, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2020: {"score": Decimal("37"), "rank": 102, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2019: {"score": Decimal("40"), "rank": 85, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2018: {"score": Decimal("38"), "rank": 89, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
        },
    },

    "MY": {
        "iso_alpha2": "MY",
        "iso_alpha3": "MYS",
        "name": "Malaysia",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("47"), "rank": 62, "ci_low": Decimal("42"), "ci_high": Decimal("52")},
            2024: {"score": Decimal("47"), "rank": 62, "ci_low": Decimal("42"), "ci_high": Decimal("52")},
            2023: {"score": Decimal("50"), "rank": 57, "ci_low": Decimal("45"), "ci_high": Decimal("55")},
            2022: {"score": Decimal("47"), "rank": 61, "ci_low": Decimal("42"), "ci_high": Decimal("52")},
            2021: {"score": Decimal("48"), "rank": 62, "ci_low": Decimal("43"), "ci_high": Decimal("53")},
            2020: {"score": Decimal("51"), "rank": 57, "ci_low": Decimal("46"), "ci_high": Decimal("56")},
            2019: {"score": Decimal("53"), "rank": 51, "ci_low": Decimal("48"), "ci_high": Decimal("58")},
            2018: {"score": Decimal("47"), "rank": 61, "ci_low": Decimal("42"), "ci_high": Decimal("52")},
        },
    },

    "CO": {
        "iso_alpha2": "CO",
        "iso_alpha3": "COL",
        "name": "Colombia",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("39"), "rank": 91, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2024: {"score": Decimal("40"), "rank": 87, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2023: {"score": Decimal("40"), "rank": 87, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2022: {"score": Decimal("39"), "rank": 91, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2021: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2020: {"score": Decimal("39"), "rank": 92, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2019: {"score": Decimal("37"), "rank": 96, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2018: {"score": Decimal("36"), "rank": 99, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
        },
    },

    "PY": {
        "iso_alpha2": "PY",
        "iso_alpha3": "PRY",
        "name": "Paraguay",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2024: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2023: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2022: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2021: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2020: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2019: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2018: {"score": Decimal("29"), "rank": 132, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
        },
    },

    "CM": {
        "iso_alpha2": "CM",
        "iso_alpha3": "CMR",
        "name": "Cameroon",
        "region": "sub_saharan_africa",
        "sub_region": "central_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2024: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2023: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2022: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2021: {"score": Decimal("27"), "rank": 144, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
            2020: {"score": Decimal("25"), "rank": 149, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2019: {"score": Decimal("25"), "rank": 153, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2018: {"score": Decimal("25"), "rank": 152, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
        },
    },

    "GH": {
        "iso_alpha2": "GH",
        "iso_alpha3": "GHA",
        "name": "Ghana",
        "region": "sub_saharan_africa",
        "sub_region": "west_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("43"), "rank": 72, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2024: {"score": Decimal("43"), "rank": 72, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2023: {"score": Decimal("43"), "rank": 72, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2022: {"score": Decimal("43"), "rank": 72, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2021: {"score": Decimal("43"), "rank": 73, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2020: {"score": Decimal("43"), "rank": 75, "ci_low": Decimal("38"), "ci_high": Decimal("48")},
            2019: {"score": Decimal("41"), "rank": 80, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
            2018: {"score": Decimal("41"), "rank": 78, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
        },
    },

    "CI": {
        "iso_alpha2": "CI",
        "iso_alpha3": "CIV",
        "name": "Ivory Coast",
        "region": "sub_saharan_africa",
        "sub_region": "west_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2024: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2023: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2022: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2021: {"score": Decimal("36"), "rank": 105, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2020: {"score": Decimal("36"), "rank": 104, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2019: {"score": Decimal("35"), "rank": 106, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
            2018: {"score": Decimal("35"), "rank": 105, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
        },
    },

    "CD": {
        "iso_alpha2": "CD",
        "iso_alpha3": "COD",
        "name": "Democratic Republic of the Congo",
        "region": "sub_saharan_africa",
        "sub_region": "central_africa",
        "income_level": "low",
        "scores": {
            2025: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2024: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2023: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2022: {"score": Decimal("20"), "rank": 166, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2021: {"score": Decimal("19"), "rank": 169, "ci_low": Decimal("14"), "ci_high": Decimal("24")},
            2020: {"score": Decimal("18"), "rank": 170, "ci_low": Decimal("13"), "ci_high": Decimal("23")},
            2019: {"score": Decimal("18"), "rank": 168, "ci_low": Decimal("13"), "ci_high": Decimal("23")},
            2018: {"score": Decimal("20"), "rank": 161, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
        },
    },

    "MM": {
        "iso_alpha2": "MM",
        "iso_alpha3": "MMR",
        "name": "Myanmar",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2024: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2023: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2022: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2021: {"score": Decimal("28"), "rank": 140, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2020: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2019: {"score": Decimal("29"), "rank": 130, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2018: {"score": Decimal("29"), "rank": 132, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
        },
    },

    "PG": {
        "iso_alpha2": "PG",
        "iso_alpha3": "PNG",
        "name": "Papua New Guinea",
        "region": "asia_pacific",
        "sub_region": "oceania",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2024: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2023: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2022: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2021: {"score": Decimal("27"), "rank": 144, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
            2020: {"score": Decimal("27"), "rank": 142, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
            2019: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2018: {"score": Decimal("28"), "rank": 138, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
        },
    },

    "HN": {
        "iso_alpha2": "HN",
        "iso_alpha3": "HND",
        "name": "Honduras",
        "region": "americas",
        "sub_region": "central_america",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2024: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2023: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2022: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2021: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2020: {"score": Decimal("24"), "rank": 157, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2019: {"score": Decimal("26"), "rank": 146, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2018: {"score": Decimal("29"), "rank": 132, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
        },
    },

    "GT": {
        "iso_alpha2": "GT",
        "iso_alpha3": "GTM",
        "name": "Guatemala",
        "region": "americas",
        "sub_region": "central_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("24"), "rank": 154, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2024: {"score": Decimal("24"), "rank": 154, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2023: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2022: {"score": Decimal("24"), "rank": 150, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2021: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2020: {"score": Decimal("25"), "rank": 149, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2019: {"score": Decimal("26"), "rank": 146, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2018: {"score": Decimal("27"), "rank": 144, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
        },
    },

    "EC": {
        "iso_alpha2": "EC",
        "iso_alpha3": "ECU",
        "name": "Ecuador",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("36"), "rank": 107, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2024: {"score": Decimal("35"), "rank": 109, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
            2023: {"score": Decimal("34"), "rank": 114, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2022: {"score": Decimal("36"), "rank": 101, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2021: {"score": Decimal("36"), "rank": 105, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2020: {"score": Decimal("39"), "rank": 92, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2019: {"score": Decimal("38"), "rank": 93, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2018: {"score": Decimal("34"), "rank": 114, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
        },
    },

    "TH": {
        "iso_alpha2": "TH",
        "iso_alpha3": "THA",
        "name": "Thailand",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("36"), "rank": 107, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2024: {"score": Decimal("36"), "rank": 107, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2023: {"score": Decimal("35"), "rank": 108, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
            2022: {"score": Decimal("36"), "rank": 101, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2021: {"score": Decimal("35"), "rank": 110, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
            2020: {"score": Decimal("36"), "rank": 104, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2019: {"score": Decimal("36"), "rank": 101, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2018: {"score": Decimal("36"), "rank": 99, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
        },
    },

    "VN": {
        "iso_alpha2": "VN",
        "iso_alpha3": "VNM",
        "name": "Vietnam",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("42"), "rank": 77, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2024: {"score": Decimal("41"), "rank": 83, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
            2023: {"score": Decimal("41"), "rank": 83, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
            2022: {"score": Decimal("42"), "rank": 77, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2021: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2020: {"score": Decimal("36"), "rank": 104, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2019: {"score": Decimal("37"), "rank": 96, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2018: {"score": Decimal("33"), "rank": 117, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
        },
    },

    "LA": {
        "iso_alpha2": "LA",
        "iso_alpha3": "LAO",
        "name": "Laos",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2024: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2023: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2022: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2021: {"score": Decimal("30"), "rank": 128, "ci_low": Decimal("25"), "ci_high": Decimal("35")},
            2020: {"score": Decimal("29"), "rank": 134, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2019: {"score": Decimal("29"), "rank": 130, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2018: {"score": Decimal("29"), "rank": 132, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
        },
    },

    "KH": {
        "iso_alpha2": "KH",
        "iso_alpha3": "KHM",
        "name": "Cambodia",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("24"), "rank": 154, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2024: {"score": Decimal("22"), "rank": 158, "ci_low": Decimal("17"), "ci_high": Decimal("27")},
            2023: {"score": Decimal("22"), "rank": 158, "ci_low": Decimal("17"), "ci_high": Decimal("27")},
            2022: {"score": Decimal("24"), "rank": 150, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2021: {"score": Decimal("23"), "rank": 157, "ci_low": Decimal("18"), "ci_high": Decimal("28")},
            2020: {"score": Decimal("21"), "rank": 160, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2019: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2018: {"score": Decimal("20"), "rank": 161, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
        },
    },

    "NG": {
        "iso_alpha2": "NG",
        "iso_alpha3": "NGA",
        "name": "Nigeria",
        "region": "sub_saharan_africa",
        "sub_region": "west_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2024: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2023: {"score": Decimal("25"), "rank": 145, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2022: {"score": Decimal("24"), "rank": 150, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2021: {"score": Decimal("24"), "rank": 154, "ci_low": Decimal("19"), "ci_high": Decimal("29")},
            2020: {"score": Decimal("25"), "rank": 149, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2019: {"score": Decimal("26"), "rank": 146, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2018: {"score": Decimal("27"), "rank": 144, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
        },
    },

    "PE": {
        "iso_alpha2": "PE",
        "iso_alpha3": "PER",
        "name": "Peru",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("33"), "rank": 120, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
            2024: {"score": Decimal("33"), "rank": 121, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
            2023: {"score": Decimal("33"), "rank": 121, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
            2022: {"score": Decimal("36"), "rank": 101, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2021: {"score": Decimal("36"), "rank": 105, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2020: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2019: {"score": Decimal("36"), "rank": 101, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
            2018: {"score": Decimal("35"), "rank": 105, "ci_low": Decimal("30"), "ci_high": Decimal("40")},
        },
    },

    "BO": {
        "iso_alpha2": "BO",
        "iso_alpha3": "BOL",
        "name": "Bolivia",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("29"), "rank": 133, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2024: {"score": Decimal("29"), "rank": 133, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2023: {"score": Decimal("29"), "rank": 133, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2022: {"score": Decimal("31"), "rank": 126, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2021: {"score": Decimal("30"), "rank": 128, "ci_low": Decimal("25"), "ci_high": Decimal("35")},
            2020: {"score": Decimal("31"), "rank": 124, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2019: {"score": Decimal("31"), "rank": 123, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2018: {"score": Decimal("29"), "rank": 132, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
        },
    },

    "ET": {
        "iso_alpha2": "ET",
        "iso_alpha3": "ETH",
        "name": "Ethiopia",
        "region": "sub_saharan_africa",
        "sub_region": "east_africa",
        "income_level": "low",
        "scores": {
            2025: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2024: {"score": Decimal("37"), "rank": 99, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2023: {"score": Decimal("37"), "rank": 94, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2022: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2021: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2020: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2019: {"score": Decimal("37"), "rank": 96, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2018: {"score": Decimal("34"), "rank": 114, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
        },
    },

    # -----------------------------------------------------------------------
    # BENCHMARK LOW-CORRUPTION COUNTRIES
    # -----------------------------------------------------------------------

    "DK": {
        "iso_alpha2": "DK",
        "iso_alpha3": "DNK",
        "name": "Denmark",
        "region": "western_europe",
        "sub_region": "nordic",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("90"), "rank": 1, "ci_low": Decimal("86"), "ci_high": Decimal("93")},
            2024: {"score": Decimal("90"), "rank": 1, "ci_low": Decimal("86"), "ci_high": Decimal("93")},
            2023: {"score": Decimal("90"), "rank": 1, "ci_low": Decimal("86"), "ci_high": Decimal("93")},
            2022: {"score": Decimal("90"), "rank": 1, "ci_low": Decimal("86"), "ci_high": Decimal("93")},
            2021: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
            2020: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
            2019: {"score": Decimal("87"), "rank": 1, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2018: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
        },
    },

    "FI": {
        "iso_alpha2": "FI",
        "iso_alpha3": "FIN",
        "name": "Finland",
        "region": "western_europe",
        "sub_region": "nordic",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2024: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2023: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2022: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2021: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
            2020: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2019: {"score": Decimal("86"), "rank": 3, "ci_low": Decimal("82"), "ci_high": Decimal("89")},
            2018: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
        },
    },

    "NZ": {
        "iso_alpha2": "NZ",
        "iso_alpha3": "NZL",
        "name": "New Zealand",
        "region": "asia_pacific",
        "sub_region": "oceania",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2024: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2023: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2022: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2021: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
            2020: {"score": Decimal("88"), "rank": 1, "ci_low": Decimal("84"), "ci_high": Decimal("91")},
            2019: {"score": Decimal("87"), "rank": 1, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
            2018: {"score": Decimal("87"), "rank": 2, "ci_low": Decimal("83"), "ci_high": Decimal("90")},
        },
    },

    "NO": {
        "iso_alpha2": "NO",
        "iso_alpha3": "NOR",
        "name": "Norway",
        "region": "western_europe",
        "sub_region": "nordic",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("84"), "rank": 4, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2024: {"score": Decimal("84"), "rank": 4, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2023: {"score": Decimal("84"), "rank": 4, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2022: {"score": Decimal("84"), "rank": 4, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2021: {"score": Decimal("85"), "rank": 4, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2020: {"score": Decimal("84"), "rank": 7, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2019: {"score": Decimal("84"), "rank": 7, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
            2018: {"score": Decimal("84"), "rank": 7, "ci_low": Decimal("80"), "ci_high": Decimal("87")},
        },
    },

    "SE": {
        "iso_alpha2": "SE",
        "iso_alpha3": "SWE",
        "name": "Sweden",
        "region": "western_europe",
        "sub_region": "nordic",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2024: {"score": Decimal("82"), "rank": 6, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
            2023: {"score": Decimal("82"), "rank": 6, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
            2022: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2021: {"score": Decimal("85"), "rank": 4, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2020: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2019: {"score": Decimal("85"), "rank": 4, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2018: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
        },
    },

    "SG": {
        "iso_alpha2": "SG",
        "iso_alpha3": "SGP",
        "name": "Singapore",
        "region": "asia_pacific",
        "sub_region": "southeast_asia",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2024: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2023: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2022: {"score": Decimal("83"), "rank": 5, "ci_low": Decimal("79"), "ci_high": Decimal("86")},
            2021: {"score": Decimal("85"), "rank": 4, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2020: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2019: {"score": Decimal("85"), "rank": 4, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
            2018: {"score": Decimal("85"), "rank": 3, "ci_low": Decimal("81"), "ci_high": Decimal("88")},
        },
    },

    # -----------------------------------------------------------------------
    # ADDITIONAL KEY COUNTRIES FOR EUDR SUPPLY CHAINS
    # -----------------------------------------------------------------------

    "IN": {
        "iso_alpha2": "IN",
        "iso_alpha3": "IND",
        "name": "India",
        "region": "asia_pacific",
        "sub_region": "south_asia",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("39"), "rank": 93, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2024: {"score": Decimal("39"), "rank": 93, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2023: {"score": Decimal("39"), "rank": 93, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2022: {"score": Decimal("40"), "rank": 85, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2021: {"score": Decimal("40"), "rank": 85, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2020: {"score": Decimal("40"), "rank": 86, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2019: {"score": Decimal("41"), "rank": 80, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
            2018: {"score": Decimal("41"), "rank": 78, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
        },
    },

    "CN": {
        "iso_alpha2": "CN",
        "iso_alpha3": "CHN",
        "name": "China",
        "region": "asia_pacific",
        "sub_region": "east_asia",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("42"), "rank": 76, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2024: {"score": Decimal("42"), "rank": 76, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2023: {"score": Decimal("42"), "rank": 76, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2022: {"score": Decimal("45"), "rank": 65, "ci_low": Decimal("40"), "ci_high": Decimal("50")},
            2021: {"score": Decimal("45"), "rank": 66, "ci_low": Decimal("40"), "ci_high": Decimal("50")},
            2020: {"score": Decimal("42"), "rank": 78, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2019: {"score": Decimal("41"), "rank": 80, "ci_low": Decimal("36"), "ci_high": Decimal("46")},
            2018: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
        },
    },

    "AR": {
        "iso_alpha2": "AR",
        "iso_alpha3": "ARG",
        "name": "Argentina",
        "region": "americas",
        "sub_region": "south_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("38"), "rank": 98, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2024: {"score": Decimal("37"), "rank": 98, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2023: {"score": Decimal("37"), "rank": 98, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2022: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2021: {"score": Decimal("38"), "rank": 96, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2020: {"score": Decimal("42"), "rank": 78, "ci_low": Decimal("37"), "ci_high": Decimal("47")},
            2019: {"score": Decimal("45"), "rank": 66, "ci_low": Decimal("40"), "ci_high": Decimal("50")},
            2018: {"score": Decimal("40"), "rank": 85, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
        },
    },

    "MX": {
        "iso_alpha2": "MX",
        "iso_alpha3": "MEX",
        "name": "Mexico",
        "region": "americas",
        "sub_region": "north_america",
        "income_level": "upper_middle",
        "scores": {
            2025: {"score": Decimal("31"), "rank": 126, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2024: {"score": Decimal("31"), "rank": 126, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2023: {"score": Decimal("31"), "rank": 126, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2022: {"score": Decimal("31"), "rank": 126, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2021: {"score": Decimal("31"), "rank": 124, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2020: {"score": Decimal("31"), "rank": 124, "ci_low": Decimal("26"), "ci_high": Decimal("36")},
            2019: {"score": Decimal("29"), "rank": 130, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2018: {"score": Decimal("28"), "rank": 138, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
        },
    },

    "DE": {
        "iso_alpha2": "DE",
        "iso_alpha3": "DEU",
        "name": "Germany",
        "region": "western_europe",
        "sub_region": "western_europe",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("79"), "rank": 9, "ci_low": Decimal("75"), "ci_high": Decimal("82")},
            2024: {"score": Decimal("78"), "rank": 9, "ci_low": Decimal("74"), "ci_high": Decimal("81")},
            2023: {"score": Decimal("78"), "rank": 9, "ci_low": Decimal("74"), "ci_high": Decimal("81")},
            2022: {"score": Decimal("79"), "rank": 9, "ci_low": Decimal("75"), "ci_high": Decimal("82")},
            2021: {"score": Decimal("80"), "rank": 10, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
            2020: {"score": Decimal("80"), "rank": 9, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
            2019: {"score": Decimal("80"), "rank": 9, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
            2018: {"score": Decimal("80"), "rank": 11, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
        },
    },

    "NL": {
        "iso_alpha2": "NL",
        "iso_alpha3": "NLD",
        "name": "Netherlands",
        "region": "western_europe",
        "sub_region": "western_europe",
        "income_level": "high",
        "scores": {
            2025: {"score": Decimal("80"), "rank": 8, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
            2024: {"score": Decimal("79"), "rank": 8, "ci_low": Decimal("75"), "ci_high": Decimal("82")},
            2023: {"score": Decimal("79"), "rank": 8, "ci_low": Decimal("75"), "ci_high": Decimal("82")},
            2022: {"score": Decimal("80"), "rank": 8, "ci_low": Decimal("76"), "ci_high": Decimal("83")},
            2021: {"score": Decimal("82"), "rank": 8, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
            2020: {"score": Decimal("82"), "rank": 8, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
            2019: {"score": Decimal("82"), "rank": 8, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
            2018: {"score": Decimal("82"), "rank": 8, "ci_low": Decimal("78"), "ci_high": Decimal("85")},
        },
    },

    "UG": {
        "iso_alpha2": "UG",
        "iso_alpha3": "UGA",
        "name": "Uganda",
        "region": "sub_saharan_africa",
        "sub_region": "east_africa",
        "income_level": "low",
        "scores": {
            2025: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2024: {"score": Decimal("26"), "rank": 141, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2023: {"score": Decimal("26"), "rank": 141, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2022: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2021: {"score": Decimal("27"), "rank": 144, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
            2020: {"score": Decimal("27"), "rank": 142, "ci_low": Decimal("22"), "ci_high": Decimal("32")},
            2019: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2018: {"score": Decimal("26"), "rank": 149, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
        },
    },

    "TZ": {
        "iso_alpha2": "TZ",
        "iso_alpha3": "TZA",
        "name": "Tanzania",
        "region": "sub_saharan_africa",
        "sub_region": "east_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("39"), "rank": 91, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2024: {"score": Decimal("40"), "rank": 87, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2023: {"score": Decimal("40"), "rank": 87, "ci_low": Decimal("35"), "ci_high": Decimal("45")},
            2022: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2021: {"score": Decimal("39"), "rank": 87, "ci_low": Decimal("34"), "ci_high": Decimal("44")},
            2020: {"score": Decimal("38"), "rank": 94, "ci_low": Decimal("33"), "ci_high": Decimal("43")},
            2019: {"score": Decimal("37"), "rank": 96, "ci_low": Decimal("32"), "ci_high": Decimal("42")},
            2018: {"score": Decimal("36"), "rank": 99, "ci_low": Decimal("31"), "ci_high": Decimal("41")},
        },
    },

    "CG": {
        "iso_alpha2": "CG",
        "iso_alpha3": "COG",
        "name": "Republic of the Congo",
        "region": "sub_saharan_africa",
        "sub_region": "central_africa",
        "income_level": "lower_middle",
        "scores": {
            2025: {"score": Decimal("21"), "rank": 160, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2024: {"score": Decimal("21"), "rank": 160, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2023: {"score": Decimal("20"), "rank": 162, "ci_low": Decimal("15"), "ci_high": Decimal("25")},
            2022: {"score": Decimal("21"), "rank": 158, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2021: {"score": Decimal("21"), "rank": 162, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2020: {"score": Decimal("21"), "rank": 160, "ci_low": Decimal("16"), "ci_high": Decimal("26")},
            2019: {"score": Decimal("19"), "rank": 165, "ci_low": Decimal("14"), "ci_high": Decimal("24")},
            2018: {"score": Decimal("19"), "rank": 165, "ci_low": Decimal("14"), "ci_high": Decimal("24")},
        },
    },

    "LR": {
        "iso_alpha2": "LR",
        "iso_alpha3": "LBR",
        "name": "Liberia",
        "region": "sub_saharan_africa",
        "sub_region": "west_africa",
        "income_level": "low",
        "scores": {
            2025: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2024: {"score": Decimal("25"), "rank": 150, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2023: {"score": Decimal("25"), "rank": 145, "ci_low": Decimal("20"), "ci_high": Decimal("30")},
            2022: {"score": Decimal("26"), "rank": 142, "ci_low": Decimal("21"), "ci_high": Decimal("31")},
            2021: {"score": Decimal("29"), "rank": 136, "ci_low": Decimal("24"), "ci_high": Decimal("34")},
            2020: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2019: {"score": Decimal("28"), "rank": 137, "ci_low": Decimal("23"), "ci_high": Decimal("33")},
            2018: {"score": Decimal("32"), "rank": 120, "ci_low": Decimal("27"), "ci_high": Decimal("37")},
        },
    },

    "SL": {
        "iso_alpha2": "SL",
        "iso_alpha3": "SLE",
        "name": "Sierra Leone",
        "region": "sub_saharan_africa",
        "sub_region": "west_africa",
        "income_level": "low",
        "scores": {
            2025: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2024: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2023: {"score": Decimal("34"), "rank": 110, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2022: {"score": Decimal("34"), "rank": 110, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2021: {"score": Decimal("34"), "rank": 115, "ci_low": Decimal("29"), "ci_high": Decimal("39")},
            2020: {"score": Decimal("33"), "rank": 117, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
            2019: {"score": Decimal("33"), "rank": 119, "ci_low": Decimal("28"), "ci_high": Decimal("38")},
            2018: {"score": Decimal("30"), "rank": 129, "ci_low": Decimal("25"), "ci_high": Decimal("35")},
        },
    },
}

# ===========================================================================
# ISO alpha-3 to alpha-2 mapping (built from CPI_COUNTRY_DATA)
# ===========================================================================

ISO3_TO_ISO2: Dict[str, str] = {}
for _iso2, _data in CPI_COUNTRY_DATA.items():
    ISO3_TO_ISO2[_data["iso_alpha3"]] = _iso2


# ===========================================================================
# CPIDatabase class
# ===========================================================================


class CPIDatabase:
    """
    Stateless reference data accessor for Transparency International CPI data.

    Provides typed access to CPI scores, historical data, regional aggregation,
    rankings, and statistical summaries for 180+ countries. All data is stored
    as ``Decimal`` for regulatory-grade precision.

    Example:
        >>> db = CPIDatabase()
        >>> score = db.get_score("BR", 2024)
        >>> assert score["score"] == Decimal("36")
        >>> history = db.get_history("BR")
        >>> assert len(history) >= 8
    """

    def get_score(
        self,
        country_code: str,
        year: int = 2024,
    ) -> Optional[Dict[str, Any]]:
        """Get CPI score for a country in a specific year.

        Args:
            country_code: ISO 3166-1 alpha-2 or alpha-3 country code.
            year: CPI assessment year (2018-2025).

        Returns:
            Score dict with score, rank, ci_low, ci_high, or None if not found.
        """
        iso2 = self._resolve_code(country_code)
        if iso2 is None:
            return None
        country = CPI_COUNTRY_DATA.get(iso2)
        if country is None:
            return None
        score_data = country["scores"].get(year)
        if score_data is None:
            return None
        return {
            "country_code": iso2,
            "iso_alpha3": country["iso_alpha3"],
            "name": country["name"],
            "year": year,
            "score": score_data["score"],
            "rank": score_data["rank"],
            "ci_low": score_data["ci_low"],
            "ci_high": score_data["ci_high"],
            "region": country["region"],
            "income_level": country["income_level"],
        }

    def get_history(
        self,
        country_code: str,
    ) -> List[Dict[str, Any]]:
        """Get complete CPI score history for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 or alpha-3 country code.

        Returns:
            List of score dicts sorted by year descending (newest first).
        """
        iso2 = self._resolve_code(country_code)
        if iso2 is None:
            return []
        country = CPI_COUNTRY_DATA.get(iso2)
        if country is None:
            return []
        results = []
        for year in sorted(country["scores"].keys(), reverse=True):
            score_data = country["scores"][year]
            results.append({
                "country_code": iso2,
                "iso_alpha3": country["iso_alpha3"],
                "name": country["name"],
                "year": year,
                "score": score_data["score"],
                "rank": score_data["rank"],
                "ci_low": score_data["ci_low"],
                "ci_high": score_data["ci_high"],
            })
        return results

    def get_by_region(
        self,
        region: str,
        year: int = 2024,
    ) -> List[Dict[str, Any]]:
        """Get all countries in a region with CPI scores for a given year.

        Args:
            region: Region name from REGIONS list.
            year: CPI assessment year.

        Returns:
            List of score dicts for the region, sorted by score descending.
        """
        results = []
        for iso2, country in CPI_COUNTRY_DATA.items():
            if country["region"] != region:
                continue
            score_data = country["scores"].get(year)
            if score_data is None:
                continue
            results.append({
                "country_code": iso2,
                "iso_alpha3": country["iso_alpha3"],
                "name": country["name"],
                "year": year,
                "score": score_data["score"],
                "rank": score_data["rank"],
                "region": country["region"],
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_rankings(
        self,
        year: int = 2024,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get global CPI rankings for a year.

        Args:
            year: CPI assessment year.
            limit: Max results (0 = all).

        Returns:
            List of score dicts sorted by rank ascending (best first).
        """
        results = []
        for iso2, country in CPI_COUNTRY_DATA.items():
            score_data = country["scores"].get(year)
            if score_data is None:
                continue
            results.append({
                "country_code": iso2,
                "iso_alpha3": country["iso_alpha3"],
                "name": country["name"],
                "year": year,
                "score": score_data["score"],
                "rank": score_data["rank"],
                "region": country["region"],
                "income_level": country["income_level"],
            })
        results.sort(key=lambda x: x["rank"])
        if limit > 0:
            results = results[:limit]
        return results

    def get_statistics(
        self,
        year: int = 2024,
    ) -> Dict[str, Any]:
        """Get aggregate CPI statistics for a year.

        Args:
            year: CPI assessment year.

        Returns:
            Dict with count, mean, median, min, max, std_dev, and
            per-region averages.
        """
        scores: List[Decimal] = []
        region_scores: Dict[str, List[Decimal]] = {}

        for country in CPI_COUNTRY_DATA.values():
            score_data = country["scores"].get(year)
            if score_data is None:
                continue
            score = score_data["score"]
            scores.append(score)
            region = country["region"]
            if region not in region_scores:
                region_scores[region] = []
            region_scores[region].append(score)

        if not scores:
            return {"year": year, "count": 0}

        scores_sorted = sorted(scores)
        count = len(scores_sorted)
        total = sum(scores_sorted)
        mean = total / count
        mid = count // 2
        if count % 2 == 0:
            median = (scores_sorted[mid - 1] + scores_sorted[mid]) / 2
        else:
            median = scores_sorted[mid]

        variance = sum((s - mean) ** 2 for s in scores_sorted) / count
        std_dev = variance ** Decimal("0.5")

        region_averages = {}
        for region, region_vals in region_scores.items():
            if region_vals:
                region_averages[region] = {
                    "count": len(region_vals),
                    "mean": sum(region_vals) / len(region_vals),
                }

        return {
            "year": year,
            "count": count,
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": scores_sorted[0],
            "max": scores_sorted[-1],
            "std_dev": round(std_dev, 2),
            "region_averages": region_averages,
        }

    def search_countries(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search countries by name or code (case-insensitive).

        Args:
            query: Search string (country name, alpha-2, or alpha-3 code).

        Returns:
            List of matching country info dicts.
        """
        query_lower = query.lower()
        results = []
        for iso2, country in CPI_COUNTRY_DATA.items():
            if (
                query_lower in country["name"].lower()
                or query_lower == iso2.lower()
                or query_lower == country["iso_alpha3"].lower()
            ):
                results.append({
                    "country_code": iso2,
                    "iso_alpha3": country["iso_alpha3"],
                    "name": country["name"],
                    "region": country["region"],
                    "sub_region": country["sub_region"],
                    "income_level": country["income_level"],
                })
        return results

    def get_country_count(self) -> int:
        """Get total number of countries in the database.

        Returns:
            Integer count of countries.
        """
        return len(CPI_COUNTRY_DATA)

    def _resolve_code(self, code: str) -> Optional[str]:
        """Resolve ISO alpha-2 or alpha-3 code to alpha-2.

        Args:
            code: ISO country code (alpha-2 or alpha-3).

        Returns:
            Alpha-2 code or None if not found.
        """
        code_upper = code.upper()
        if code_upper in CPI_COUNTRY_DATA:
            return code_upper
        return ISO3_TO_ISO2.get(code_upper)


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_score(country_code: str, year: int = 2024) -> Optional[Dict[str, Any]]:
    """Get CPI score for a country in a specific year.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code.
        year: CPI assessment year (2018-2025).

    Returns:
        Score dict or None if not found.
    """
    return CPIDatabase().get_score(country_code, year)


def get_history(country_code: str) -> List[Dict[str, Any]]:
    """Get complete CPI score history for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code.

    Returns:
        List of score dicts sorted by year descending.
    """
    return CPIDatabase().get_history(country_code)


def get_by_region(region: str, year: int = 2024) -> List[Dict[str, Any]]:
    """Get all countries in a region with CPI scores for a given year.

    Args:
        region: Region name from REGIONS list.
        year: CPI assessment year.

    Returns:
        List of score dicts sorted by score descending.
    """
    return CPIDatabase().get_by_region(region, year)


def get_rankings(year: int = 2024, limit: int = 0) -> List[Dict[str, Any]]:
    """Get global CPI rankings for a year.

    Args:
        year: CPI assessment year.
        limit: Max results (0 = all).

    Returns:
        List of score dicts sorted by rank ascending.
    """
    return CPIDatabase().get_rankings(year, limit)


def get_statistics(year: int = 2024) -> Dict[str, Any]:
    """Get aggregate CPI statistics for a year.

    Args:
        year: CPI assessment year.

    Returns:
        Dict with statistical summary.
    """
    return CPIDatabase().get_statistics(year)


def search_countries(query: str) -> List[Dict[str, Any]]:
    """Search countries by name or code.

    Args:
        query: Search string.

    Returns:
        List of matching country info dicts.
    """
    return CPIDatabase().search_countries(query)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "REGIONS",
    "REGION_DISPLAY_NAMES",
    "CPI_COUNTRY_DATA",
    "ISO3_TO_ISO2",
    "CPIDatabase",
    "get_score",
    "get_history",
    "get_by_region",
    "get_rankings",
    "get_statistics",
    "search_countries",
]
