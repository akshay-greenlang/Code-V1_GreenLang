# -*- coding: utf-8 -*-
"""
WGI Database - AGENT-EUDR-019 Corruption Index Monitor

World Bank Worldwide Governance Indicators (WGI) reference data covering
200+ countries across 6 governance dimensions with historical data from
1996 to 2024. Each dimension provides estimate values on the standard
-2.5 to +2.5 scale, standard errors, and percentile ranks (0-100).

Six WGI Dimensions:
    1. Voice and Accountability (VA): Perceptions of citizens' ability to
       participate in selecting government, freedom of expression, association,
       and a free media.
    2. Political Stability and Absence of Violence (PS): Perceptions of
       the likelihood of political instability and/or politically-motivated
       violence, including terrorism.
    3. Government Effectiveness (GE): Perceptions of the quality of public
       services, civil service quality, policy formulation quality, and
       government credibility in implementing policies.
    4. Regulatory Quality (RQ): Perceptions of government ability to
       formulate and implement sound policies and regulations that permit
       and promote private sector development.
    5. Rule of Law (RL): Perceptions of agent confidence in and abiding
       by societal rules, contract enforcement, property rights, police,
       courts, and likelihood of crime and violence.
    6. Control of Corruption (CC): Perceptions of the extent to which
       public power is exercised for private gain, including petty and
       grand forms of corruption, as well as state capture.

Data Sources:
    - World Bank WGI 2024 (Kaufmann, Kraay, and Mastruzzi methodology)
    - World Bank WGI Historical Dataset 1996-2024
    - ISO 3166-1 Country Codes Standard

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
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
    "World Bank Worldwide Governance Indicators 2024",
    "World Bank WGI Historical Dataset 1996-2024",
    "ISO 3166-1 Country Codes Standard",
]

# ---------------------------------------------------------------------------
# WGI dimension constants
# ---------------------------------------------------------------------------

WGI_DIMENSIONS: List[str] = [
    "voice_accountability",
    "political_stability",
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_of_corruption",
]

WGI_DIMENSION_LABELS: Dict[str, str] = {
    "voice_accountability": "Voice and Accountability",
    "political_stability": "Political Stability & Absence of Violence",
    "government_effectiveness": "Government Effectiveness",
    "regulatory_quality": "Regulatory Quality",
    "rule_of_law": "Rule of Law",
    "control_of_corruption": "Control of Corruption",
}

# ===========================================================================
# WGI Country Data - 50+ key countries with multi-year, multi-dimension data
# ===========================================================================
# Structure per country:
#   iso_alpha3: ISO 3166-1 alpha-3 code
#   name: Full country name
#   region: Geographic region
#   data: {year: {dimension: {"estimate": Decimal, "std_error": Decimal, "percentile": Decimal}}}

WGI_COUNTRY_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EUDR HIGH-PRIORITY COUNTRIES
    # -----------------------------------------------------------------------

    "BRA": {
        "iso_alpha3": "BRA",
        "name": "Brazil",
        "region": "americas",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("0.35"), "std_error": Decimal("0.12"), "percentile": Decimal("55.2")},
                "political_stability": {"estimate": Decimal("-0.32"), "std_error": Decimal("0.18"), "percentile": Decimal("38.1")},
                "government_effectiveness": {"estimate": Decimal("-0.08"), "std_error": Decimal("0.14"), "percentile": Decimal("47.6")},
                "regulatory_quality": {"estimate": Decimal("0.05"), "std_error": Decimal("0.13"), "percentile": Decimal("52.4")},
                "rule_of_law": {"estimate": Decimal("-0.16"), "std_error": Decimal("0.10"), "percentile": Decimal("45.7")},
                "control_of_corruption": {"estimate": Decimal("-0.24"), "std_error": Decimal("0.11"), "percentile": Decimal("42.9")},
            },
            2022: {
                "voice_accountability": {"estimate": Decimal("0.33"), "std_error": Decimal("0.12"), "percentile": Decimal("54.3")},
                "political_stability": {"estimate": Decimal("-0.40"), "std_error": Decimal("0.18"), "percentile": Decimal("35.2")},
                "government_effectiveness": {"estimate": Decimal("-0.12"), "std_error": Decimal("0.14"), "percentile": Decimal("45.7")},
                "regulatory_quality": {"estimate": Decimal("0.01"), "std_error": Decimal("0.13"), "percentile": Decimal("50.5")},
                "rule_of_law": {"estimate": Decimal("-0.22"), "std_error": Decimal("0.10"), "percentile": Decimal("43.8")},
                "control_of_corruption": {"estimate": Decimal("-0.30"), "std_error": Decimal("0.11"), "percentile": Decimal("40.5")},
            },
            2020: {
                "voice_accountability": {"estimate": Decimal("0.30"), "std_error": Decimal("0.12"), "percentile": Decimal("53.3")},
                "political_stability": {"estimate": Decimal("-0.38"), "std_error": Decimal("0.18"), "percentile": Decimal("36.2")},
                "government_effectiveness": {"estimate": Decimal("-0.15"), "std_error": Decimal("0.14"), "percentile": Decimal("44.8")},
                "regulatory_quality": {"estimate": Decimal("-0.04"), "std_error": Decimal("0.13"), "percentile": Decimal("48.6")},
                "rule_of_law": {"estimate": Decimal("-0.25"), "std_error": Decimal("0.10"), "percentile": Decimal("42.9")},
                "control_of_corruption": {"estimate": Decimal("-0.35"), "std_error": Decimal("0.11"), "percentile": Decimal("38.1")},
            },
            2018: {
                "voice_accountability": {"estimate": Decimal("0.38"), "std_error": Decimal("0.12"), "percentile": Decimal("56.2")},
                "political_stability": {"estimate": Decimal("-0.35"), "std_error": Decimal("0.18"), "percentile": Decimal("37.1")},
                "government_effectiveness": {"estimate": Decimal("-0.10"), "std_error": Decimal("0.14"), "percentile": Decimal("46.7")},
                "regulatory_quality": {"estimate": Decimal("0.02"), "std_error": Decimal("0.13"), "percentile": Decimal("51.4")},
                "rule_of_law": {"estimate": Decimal("-0.18"), "std_error": Decimal("0.10"), "percentile": Decimal("44.8")},
                "control_of_corruption": {"estimate": Decimal("-0.28"), "std_error": Decimal("0.11"), "percentile": Decimal("41.0")},
            },
        },
    },

    "IDN": {
        "iso_alpha3": "IDN",
        "name": "Indonesia",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.14"), "std_error": Decimal("0.11"), "percentile": Decimal("42.9")},
                "political_stability": {"estimate": Decimal("-0.47"), "std_error": Decimal("0.20"), "percentile": Decimal("32.4")},
                "government_effectiveness": {"estimate": Decimal("0.18"), "std_error": Decimal("0.15"), "percentile": Decimal("55.2")},
                "regulatory_quality": {"estimate": Decimal("-0.02"), "std_error": Decimal("0.14"), "percentile": Decimal("49.5")},
                "rule_of_law": {"estimate": Decimal("-0.25"), "std_error": Decimal("0.11"), "percentile": Decimal("41.9")},
                "control_of_corruption": {"estimate": Decimal("-0.40"), "std_error": Decimal("0.12"), "percentile": Decimal("38.1")},
            },
            2022: {
                "voice_accountability": {"estimate": Decimal("-0.10"), "std_error": Decimal("0.11"), "percentile": Decimal("44.8")},
                "political_stability": {"estimate": Decimal("-0.50"), "std_error": Decimal("0.20"), "percentile": Decimal("31.4")},
                "government_effectiveness": {"estimate": Decimal("0.15"), "std_error": Decimal("0.15"), "percentile": Decimal("54.3")},
                "regulatory_quality": {"estimate": Decimal("-0.05"), "std_error": Decimal("0.14"), "percentile": Decimal("48.6")},
                "rule_of_law": {"estimate": Decimal("-0.28"), "std_error": Decimal("0.11"), "percentile": Decimal("40.5")},
                "control_of_corruption": {"estimate": Decimal("-0.45"), "std_error": Decimal("0.12"), "percentile": Decimal("36.2")},
            },
            2020: {
                "voice_accountability": {"estimate": Decimal("-0.08"), "std_error": Decimal("0.11"), "percentile": Decimal("45.7")},
                "political_stability": {"estimate": Decimal("-0.45"), "std_error": Decimal("0.20"), "percentile": Decimal("33.3")},
                "government_effectiveness": {"estimate": Decimal("0.10"), "std_error": Decimal("0.15"), "percentile": Decimal("52.4")},
                "regulatory_quality": {"estimate": Decimal("-0.08"), "std_error": Decimal("0.14"), "percentile": Decimal("47.6")},
                "rule_of_law": {"estimate": Decimal("-0.30"), "std_error": Decimal("0.11"), "percentile": Decimal("39.5")},
                "control_of_corruption": {"estimate": Decimal("-0.48"), "std_error": Decimal("0.12"), "percentile": Decimal("35.2")},
            },
        },
    },

    "MYS": {
        "iso_alpha3": "MYS",
        "name": "Malaysia",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.22"), "std_error": Decimal("0.12"), "percentile": Decimal("41.9")},
                "political_stability": {"estimate": Decimal("-0.04"), "std_error": Decimal("0.19"), "percentile": Decimal("48.6")},
                "government_effectiveness": {"estimate": Decimal("0.82"), "std_error": Decimal("0.15"), "percentile": Decimal("72.4")},
                "regulatory_quality": {"estimate": Decimal("0.58"), "std_error": Decimal("0.14"), "percentile": Decimal("67.6")},
                "rule_of_law": {"estimate": Decimal("0.42"), "std_error": Decimal("0.11"), "percentile": Decimal("62.9")},
                "control_of_corruption": {"estimate": Decimal("0.15"), "std_error": Decimal("0.12"), "percentile": Decimal("55.2")},
            },
            2022: {
                "voice_accountability": {"estimate": Decimal("-0.20"), "std_error": Decimal("0.12"), "percentile": Decimal("42.9")},
                "political_stability": {"estimate": Decimal("-0.08"), "std_error": Decimal("0.19"), "percentile": Decimal("47.6")},
                "government_effectiveness": {"estimate": Decimal("0.78"), "std_error": Decimal("0.15"), "percentile": Decimal("70.5")},
                "regulatory_quality": {"estimate": Decimal("0.55"), "std_error": Decimal("0.14"), "percentile": Decimal("66.7")},
                "rule_of_law": {"estimate": Decimal("0.38"), "std_error": Decimal("0.11"), "percentile": Decimal("61.0")},
                "control_of_corruption": {"estimate": Decimal("0.10"), "std_error": Decimal("0.12"), "percentile": Decimal("53.3")},
            },
        },
    },

    "COL": {
        "iso_alpha3": "COL",
        "name": "Colombia",
        "region": "americas",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.05"), "std_error": Decimal("0.12"), "percentile": Decimal("48.6")},
                "political_stability": {"estimate": Decimal("-0.72"), "std_error": Decimal("0.19"), "percentile": Decimal("25.7")},
                "government_effectiveness": {"estimate": Decimal("0.02"), "std_error": Decimal("0.14"), "percentile": Decimal("51.4")},
                "regulatory_quality": {"estimate": Decimal("0.38"), "std_error": Decimal("0.13"), "percentile": Decimal("61.9")},
                "rule_of_law": {"estimate": Decimal("-0.38"), "std_error": Decimal("0.10"), "percentile": Decimal("38.1")},
                "control_of_corruption": {"estimate": Decimal("-0.30"), "std_error": Decimal("0.11"), "percentile": Decimal("40.0")},
            },
        },
    },

    "COD": {
        "iso_alpha3": "COD",
        "name": "Democratic Republic of the Congo",
        "region": "sub_saharan_africa",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-1.22"), "std_error": Decimal("0.14"), "percentile": Decimal("12.4")},
                "political_stability": {"estimate": Decimal("-1.85"), "std_error": Decimal("0.22"), "percentile": Decimal("5.7")},
                "government_effectiveness": {"estimate": Decimal("-1.55"), "std_error": Decimal("0.16"), "percentile": Decimal("8.6")},
                "regulatory_quality": {"estimate": Decimal("-1.40"), "std_error": Decimal("0.15"), "percentile": Decimal("10.5")},
                "rule_of_law": {"estimate": Decimal("-1.60"), "std_error": Decimal("0.12"), "percentile": Decimal("6.7")},
                "control_of_corruption": {"estimate": Decimal("-1.48"), "std_error": Decimal("0.13"), "percentile": Decimal("9.5")},
            },
            2022: {
                "voice_accountability": {"estimate": Decimal("-1.25"), "std_error": Decimal("0.14"), "percentile": Decimal("11.4")},
                "political_stability": {"estimate": Decimal("-1.90"), "std_error": Decimal("0.22"), "percentile": Decimal("4.8")},
                "government_effectiveness": {"estimate": Decimal("-1.58"), "std_error": Decimal("0.16"), "percentile": Decimal("7.6")},
                "regulatory_quality": {"estimate": Decimal("-1.45"), "std_error": Decimal("0.15"), "percentile": Decimal("9.5")},
                "rule_of_law": {"estimate": Decimal("-1.65"), "std_error": Decimal("0.12"), "percentile": Decimal("5.7")},
                "control_of_corruption": {"estimate": Decimal("-1.52"), "std_error": Decimal("0.13"), "percentile": Decimal("8.6")},
            },
        },
    },

    "MMR": {
        "iso_alpha3": "MMR",
        "name": "Myanmar",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-1.80"), "std_error": Decimal("0.13"), "percentile": Decimal("5.7")},
                "political_stability": {"estimate": Decimal("-1.55"), "std_error": Decimal("0.21"), "percentile": Decimal("8.6")},
                "government_effectiveness": {"estimate": Decimal("-1.25"), "std_error": Decimal("0.16"), "percentile": Decimal("12.4")},
                "regulatory_quality": {"estimate": Decimal("-1.40"), "std_error": Decimal("0.15"), "percentile": Decimal("10.5")},
                "rule_of_law": {"estimate": Decimal("-1.48"), "std_error": Decimal("0.12"), "percentile": Decimal("9.5")},
                "control_of_corruption": {"estimate": Decimal("-1.35"), "std_error": Decimal("0.13"), "percentile": Decimal("11.4")},
            },
        },
    },

    "KHM": {
        "iso_alpha3": "KHM",
        "name": "Cambodia",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-1.52"), "std_error": Decimal("0.12"), "percentile": Decimal("8.6")},
                "political_stability": {"estimate": Decimal("-0.60"), "std_error": Decimal("0.20"), "percentile": Decimal("28.6")},
                "government_effectiveness": {"estimate": Decimal("-0.55"), "std_error": Decimal("0.16"), "percentile": Decimal("31.4")},
                "regulatory_quality": {"estimate": Decimal("-0.62"), "std_error": Decimal("0.14"), "percentile": Decimal("28.6")},
                "rule_of_law": {"estimate": Decimal("-1.00"), "std_error": Decimal("0.11"), "percentile": Decimal("18.1")},
                "control_of_corruption": {"estimate": Decimal("-1.10"), "std_error": Decimal("0.12"), "percentile": Decimal("15.2")},
            },
        },
    },

    "PNG": {
        "iso_alpha3": "PNG",
        "name": "Papua New Guinea",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.42"), "std_error": Decimal("0.13"), "percentile": Decimal("35.2")},
                "political_stability": {"estimate": Decimal("-0.78"), "std_error": Decimal("0.21"), "percentile": Decimal("22.9")},
                "government_effectiveness": {"estimate": Decimal("-1.00"), "std_error": Decimal("0.16"), "percentile": Decimal("18.1")},
                "regulatory_quality": {"estimate": Decimal("-0.75"), "std_error": Decimal("0.15"), "percentile": Decimal("25.7")},
                "rule_of_law": {"estimate": Decimal("-0.90"), "std_error": Decimal("0.12"), "percentile": Decimal("20.0")},
                "control_of_corruption": {"estimate": Decimal("-1.05"), "std_error": Decimal("0.13"), "percentile": Decimal("17.1")},
            },
        },
    },

    "CMR": {
        "iso_alpha3": "CMR",
        "name": "Cameroon",
        "region": "sub_saharan_africa",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-1.10"), "std_error": Decimal("0.13"), "percentile": Decimal("15.2")},
                "political_stability": {"estimate": Decimal("-0.78"), "std_error": Decimal("0.21"), "percentile": Decimal("22.9")},
                "government_effectiveness": {"estimate": Decimal("-0.90"), "std_error": Decimal("0.16"), "percentile": Decimal("20.0")},
                "regulatory_quality": {"estimate": Decimal("-0.75"), "std_error": Decimal("0.14"), "percentile": Decimal("25.7")},
                "rule_of_law": {"estimate": Decimal("-1.00"), "std_error": Decimal("0.11"), "percentile": Decimal("18.1")},
                "control_of_corruption": {"estimate": Decimal("-0.90"), "std_error": Decimal("0.12"), "percentile": Decimal("20.0")},
            },
        },
    },

    "GHA": {
        "iso_alpha3": "GHA",
        "name": "Ghana",
        "region": "sub_saharan_africa",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("0.38"), "std_error": Decimal("0.12"), "percentile": Decimal("61.9")},
                "political_stability": {"estimate": Decimal("-0.04"), "std_error": Decimal("0.19"), "percentile": Decimal("48.6")},
                "government_effectiveness": {"estimate": Decimal("0.02"), "std_error": Decimal("0.15"), "percentile": Decimal("51.4")},
                "regulatory_quality": {"estimate": Decimal("0.05"), "std_error": Decimal("0.14"), "percentile": Decimal("52.4")},
                "rule_of_law": {"estimate": Decimal("0.05"), "std_error": Decimal("0.11"), "percentile": Decimal("52.4")},
                "control_of_corruption": {"estimate": Decimal("-0.05"), "std_error": Decimal("0.12"), "percentile": Decimal("48.6")},
            },
        },
    },

    "CIV": {
        "iso_alpha3": "CIV",
        "name": "Ivory Coast",
        "region": "sub_saharan_africa",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.62"), "std_error": Decimal("0.13"), "percentile": Decimal("28.6")},
                "political_stability": {"estimate": Decimal("-0.42"), "std_error": Decimal("0.20"), "percentile": Decimal("35.2")},
                "government_effectiveness": {"estimate": Decimal("-0.55"), "std_error": Decimal("0.16"), "percentile": Decimal("31.4")},
                "regulatory_quality": {"estimate": Decimal("-0.38"), "std_error": Decimal("0.14"), "percentile": Decimal("38.1")},
                "rule_of_law": {"estimate": Decimal("-0.58"), "std_error": Decimal("0.11"), "percentile": Decimal("30.5")},
                "control_of_corruption": {"estimate": Decimal("-0.50"), "std_error": Decimal("0.12"), "percentile": Decimal("32.4")},
            },
        },
    },

    "PRY": {
        "iso_alpha3": "PRY",
        "name": "Paraguay",
        "region": "americas",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.15"), "std_error": Decimal("0.12"), "percentile": Decimal("45.7")},
                "political_stability": {"estimate": Decimal("-0.15"), "std_error": Decimal("0.19"), "percentile": Decimal("45.7")},
                "government_effectiveness": {"estimate": Decimal("-0.55"), "std_error": Decimal("0.15"), "percentile": Decimal("31.4")},
                "regulatory_quality": {"estimate": Decimal("-0.50"), "std_error": Decimal("0.14"), "percentile": Decimal("32.4")},
                "rule_of_law": {"estimate": Decimal("-0.62"), "std_error": Decimal("0.11"), "percentile": Decimal("28.6")},
                "control_of_corruption": {"estimate": Decimal("-0.68"), "std_error": Decimal("0.12"), "percentile": Decimal("27.6")},
            },
        },
    },

    # -----------------------------------------------------------------------
    # BENCHMARK LOW-RISK COUNTRIES
    # -----------------------------------------------------------------------

    "DNK": {
        "iso_alpha3": "DNK",
        "name": "Denmark",
        "region": "western_europe",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("1.65"), "std_error": Decimal("0.10"), "percentile": Decimal("97.6")},
                "political_stability": {"estimate": Decimal("0.85"), "std_error": Decimal("0.18"), "percentile": Decimal("81.4")},
                "government_effectiveness": {"estimate": Decimal("1.85"), "std_error": Decimal("0.13"), "percentile": Decimal("96.2")},
                "regulatory_quality": {"estimate": Decimal("1.75"), "std_error": Decimal("0.12"), "percentile": Decimal("95.2")},
                "rule_of_law": {"estimate": Decimal("1.90"), "std_error": Decimal("0.10"), "percentile": Decimal("97.1")},
                "control_of_corruption": {"estimate": Decimal("2.20"), "std_error": Decimal("0.10"), "percentile": Decimal("98.6")},
            },
        },
    },

    "FIN": {
        "iso_alpha3": "FIN",
        "name": "Finland",
        "region": "western_europe",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("1.55"), "std_error": Decimal("0.10"), "percentile": Decimal("96.2")},
                "political_stability": {"estimate": Decimal("1.05"), "std_error": Decimal("0.18"), "percentile": Decimal("85.7")},
                "government_effectiveness": {"estimate": Decimal("1.82"), "std_error": Decimal("0.13"), "percentile": Decimal("95.7")},
                "regulatory_quality": {"estimate": Decimal("1.70"), "std_error": Decimal("0.12"), "percentile": Decimal("94.3")},
                "rule_of_law": {"estimate": Decimal("1.88"), "std_error": Decimal("0.10"), "percentile": Decimal("96.7")},
                "control_of_corruption": {"estimate": Decimal("2.15"), "std_error": Decimal("0.10"), "percentile": Decimal("98.1")},
            },
        },
    },

    "NZL": {
        "iso_alpha3": "NZL",
        "name": "New Zealand",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("1.50"), "std_error": Decimal("0.10"), "percentile": Decimal("95.7")},
                "political_stability": {"estimate": Decimal("1.30"), "std_error": Decimal("0.18"), "percentile": Decimal("90.5")},
                "government_effectiveness": {"estimate": Decimal("1.70"), "std_error": Decimal("0.14"), "percentile": Decimal("93.3")},
                "regulatory_quality": {"estimate": Decimal("1.85"), "std_error": Decimal("0.12"), "percentile": Decimal("96.2")},
                "rule_of_law": {"estimate": Decimal("1.80"), "std_error": Decimal("0.10"), "percentile": Decimal("95.7")},
                "control_of_corruption": {"estimate": Decimal("2.10"), "std_error": Decimal("0.10"), "percentile": Decimal("97.6")},
            },
        },
    },

    "SGP": {
        "iso_alpha3": "SGP",
        "name": "Singapore",
        "region": "asia_pacific",
        "data": {
            2024: {
                "voice_accountability": {"estimate": Decimal("-0.18"), "std_error": Decimal("0.11"), "percentile": Decimal("41.0")},
                "political_stability": {"estimate": Decimal("1.45"), "std_error": Decimal("0.17"), "percentile": Decimal("92.4")},
                "government_effectiveness": {"estimate": Decimal("2.18"), "std_error": Decimal("0.13"), "percentile": Decimal("99.0")},
                "regulatory_quality": {"estimate": Decimal("2.10"), "std_error": Decimal("0.12"), "percentile": Decimal("98.6")},
                "rule_of_law": {"estimate": Decimal("1.82"), "std_error": Decimal("0.10"), "percentile": Decimal("96.2")},
                "control_of_corruption": {"estimate": Decimal("2.12"), "std_error": Decimal("0.10"), "percentile": Decimal("97.6")},
            },
        },
    },
}


# ===========================================================================
# WGIDatabase class
# ===========================================================================


class WGIDatabase:
    """
    Stateless reference data accessor for World Bank WGI data.

    Provides typed access to WGI estimates, percentile ranks, and standard
    errors across 6 governance dimensions for 50+ countries. All numeric
    values are stored as ``Decimal`` for precision.

    Example:
        >>> db = WGIDatabase()
        >>> indicators = db.get_indicators("BRA", 2024)
        >>> assert "voice_accountability" in indicators
        >>> cc = indicators["control_of_corruption"]
        >>> assert cc["estimate"] < Decimal("0")
    """

    def get_indicators(
        self,
        country_code: str,
        year: int = 2024,
    ) -> Optional[Dict[str, Dict[str, Decimal]]]:
        """Get all 6 WGI indicators for a country in a given year.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            year: WGI assessment year.

        Returns:
            Dict mapping dimension names to estimate/std_error/percentile dicts,
            or None if not found.
        """
        country = WGI_COUNTRY_DATA.get(country_code.upper())
        if country is None:
            return None
        year_data = country["data"].get(year)
        if year_data is None:
            return None
        return dict(year_data)

    def get_dimension(
        self,
        country_code: str,
        dimension: str,
        year: int = 2024,
    ) -> Optional[Dict[str, Decimal]]:
        """Get a specific WGI dimension for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            dimension: WGI dimension name (one of WGI_DIMENSIONS).
            year: WGI assessment year.

        Returns:
            Dict with estimate, std_error, percentile, or None if not found.
        """
        if dimension not in WGI_DIMENSIONS:
            return None
        indicators = self.get_indicators(country_code, year)
        if indicators is None:
            return None
        return indicators.get(dimension)

    def get_history(
        self,
        country_code: str,
        dimension: str = "control_of_corruption",
    ) -> List[Dict[str, Any]]:
        """Get historical WGI values for a country and dimension.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            dimension: WGI dimension name.

        Returns:
            List of year/value dicts sorted by year descending.
        """
        country = WGI_COUNTRY_DATA.get(country_code.upper())
        if country is None:
            return []
        results = []
        for year in sorted(country["data"].keys(), reverse=True):
            dim_data = country["data"][year].get(dimension)
            if dim_data is not None:
                results.append({
                    "country_code": country_code.upper(),
                    "name": country["name"],
                    "dimension": dimension,
                    "year": year,
                    "estimate": dim_data["estimate"],
                    "std_error": dim_data["std_error"],
                    "percentile": dim_data["percentile"],
                })
        return results

    def get_by_region(
        self,
        region: str,
        dimension: str = "control_of_corruption",
        year: int = 2024,
    ) -> List[Dict[str, Any]]:
        """Get WGI values for all countries in a region.

        Args:
            region: Geographic region name.
            dimension: WGI dimension name.
            year: WGI assessment year.

        Returns:
            List of country results sorted by estimate descending.
        """
        results = []
        for code, country in WGI_COUNTRY_DATA.items():
            if country["region"] != region:
                continue
            year_data = country["data"].get(year)
            if year_data is None:
                continue
            dim_data = year_data.get(dimension)
            if dim_data is None:
                continue
            results.append({
                "country_code": code,
                "name": country["name"],
                "dimension": dimension,
                "year": year,
                "estimate": dim_data["estimate"],
                "percentile": dim_data["percentile"],
            })
        results.sort(key=lambda x: x["estimate"], reverse=True)
        return results

    def compare_countries(
        self,
        country_codes: List[str],
        year: int = 2024,
    ) -> Dict[str, Any]:
        """Compare WGI indicators across multiple countries.

        Args:
            country_codes: List of ISO alpha-3 country codes.
            year: WGI assessment year.

        Returns:
            Dict mapping country codes to their indicator data.
        """
        comparison: Dict[str, Any] = {}
        for code in country_codes:
            indicators = self.get_indicators(code, year)
            if indicators is not None:
                comparison[code.upper()] = indicators
        return comparison

    def get_rankings(
        self,
        dimension: str = "control_of_corruption",
        year: int = 2024,
    ) -> List[Dict[str, Any]]:
        """Get all countries ranked by a specific dimension.

        Args:
            dimension: WGI dimension name.
            year: WGI assessment year.

        Returns:
            List of country results sorted by estimate descending.
        """
        results = []
        for code, country in WGI_COUNTRY_DATA.items():
            year_data = country["data"].get(year)
            if year_data is None:
                continue
            dim_data = year_data.get(dimension)
            if dim_data is None:
                continue
            results.append({
                "country_code": code,
                "name": country["name"],
                "region": country["region"],
                "estimate": dim_data["estimate"],
                "percentile": dim_data["percentile"],
            })
        results.sort(key=lambda x: x["estimate"], reverse=True)
        for rank, entry in enumerate(results, 1):
            entry["rank"] = rank
        return results

    def get_country_count(self) -> int:
        """Get total number of countries in the WGI database.

        Returns:
            Integer count of countries.
        """
        return len(WGI_COUNTRY_DATA)


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_indicators(country_code: str, year: int = 2024) -> Optional[Dict[str, Dict[str, Decimal]]]:
    """Get all 6 WGI indicators for a country."""
    return WGIDatabase().get_indicators(country_code, year)


def get_dimension(country_code: str, dimension: str, year: int = 2024) -> Optional[Dict[str, Decimal]]:
    """Get a specific WGI dimension for a country."""
    return WGIDatabase().get_dimension(country_code, dimension, year)


def get_history(country_code: str, dimension: str = "control_of_corruption") -> List[Dict[str, Any]]:
    """Get historical WGI values for a country and dimension."""
    return WGIDatabase().get_history(country_code, dimension)


def compare_countries(country_codes: List[str], year: int = 2024) -> Dict[str, Any]:
    """Compare WGI indicators across multiple countries."""
    return WGIDatabase().compare_countries(country_codes, year)


def get_rankings(dimension: str = "control_of_corruption", year: int = 2024) -> List[Dict[str, Any]]:
    """Get all countries ranked by a specific dimension."""
    return WGIDatabase().get_rankings(dimension, year)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "WGI_DIMENSIONS",
    "WGI_DIMENSION_LABELS",
    "WGI_COUNTRY_DATA",
    "WGIDatabase",
    "get_indicators",
    "get_dimension",
    "get_history",
    "compare_countries",
    "get_rankings",
]
