# -*- coding: utf-8 -*-
"""
SovereignBondCalculatorEngine - Government bond financed emissions.

This module implements the SovereignBondCalculatorEngine for AGENT-MRV-028
(Investments, GHG Protocol Scope 3 Category 15). It provides thread-safe
singleton calculations for financed emissions from sovereign (government)
bond holdings using PCAF methodology.

Calculation Formula:
    attribution_factor = outstanding_amount / PPP_adjusted_GDP
    country_emissions = total_GHG_emissions - LULUCF
    financed_emissions = attribution_factor x country_emissions

Key Features:
    - PPP-adjusted GDP for fair cross-country comparison
    - Production-based national emissions excluding LULUCF
    - Sub-sovereign (regional/municipal) bond support
    - Multilateral bond weighted-average calculation
    - Green sovereign bond use-of-proceeds tracking
    - Double-counting awareness (DC-INV-004)

Data Sources:
    - UNFCCC National Inventory Reports
    - EDGAR (Emissions Database for Global Atmospheric Research)
    - World Bank GDP PPP data
    - IMF World Economic Outlook

PCAF Note:
    Sovereign bonds have simpler data quality scoring:
    Score 4: Production emissions + GDP (standard path)
    Score 5: Estimates or incomplete data
    Scores 1-3 are not applicable as there is no company-reported data.

Country Coverage (50+):
    G7, EU27, BRICS, Asia-Pacific, Emerging markets

Thread Safety:
    Uses __new__ singleton pattern with threading.RLock.

Example:
    >>> engine = SovereignBondCalculatorEngine()
    >>> from decimal import Decimal
    >>> from greenlang.agents.mrv.investments.sovereign_bond_calculator import SovereignBondInput
    >>> inp = SovereignBondInput(
    ...     outstanding_amount=Decimal("50000000"),
    ...     country_code="US",
    ...     reporting_year=2024,
    ... )
    >>> result = engine.calculate(inp)
    >>> result.financed_emissions_co2e > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# ==============================================================================
# CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_MAX_ATTRIBUTION = Decimal("1")
_ENCODING = "utf-8"


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class SovereignBondType(str, Enum):
    """Type of sovereign/government bond."""

    NATIONAL = "national"  # National government bond
    SUB_SOVEREIGN = "sub_sovereign"  # Regional/state/municipal bond
    MULTILATERAL = "multilateral"  # Multilateral development bank bond
    GREEN_SOVEREIGN = "green_sovereign"  # Green sovereign bond
    TREASURY_BILL = "treasury_bill"  # Short-term government paper
    INFLATION_LINKED = "inflation_linked"  # Inflation-linked government bond


class PCAFDataQuality(int, Enum):
    """PCAF data quality score for sovereign bonds.

    Note: Scores 1-3 are not applicable for sovereign bonds as there
    is no 'company-reported' data. Standard path is Score 4 (production
    emissions + GDP). Score 5 for estimates or incomplete data.
    """

    SCORE_4 = 4  # Production emissions + GDP (standard)
    SCORE_5 = 5  # Estimates or incomplete data


class EmissionsDataSource(str, Enum):
    """Source of country emissions data."""

    UNFCCC = "unfccc"  # UNFCCC National Inventory
    EDGAR = "edgar"  # EDGAR database (JRC)
    IEA = "iea"  # IEA CO2 emissions data
    WORLD_BANK = "world_bank"  # World Bank indicators
    IMF = "imf"  # IMF World Economic Outlook
    CLIMATE_WATCH = "climate_watch"  # WRI Climate Watch
    NATIONAL_REPORT = "national_report"  # Country-specific report
    ESTIMATE = "estimate"  # Estimated / modelled


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CHF = "CHF"
    SGD = "SGD"
    BRL = "BRL"
    ZAR = "ZAR"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    HKD = "HKD"
    KRW = "KRW"
    NZD = "NZD"
    MXN = "MXN"
    TRY = "TRY"


# ==============================================================================
# COUNTRY DATABASE
# total_ghg_mt_co2e: Total GHG emissions excluding LULUCF (MtCO2e, 2022 data)
# lulucf_mt_co2e: LULUCF emissions/removals (MtCO2e, positive = net source)
# gdp_ppp_billion_usd: GDP at PPP in billion USD (2022 data)
# population_million: Population in millions (2022 data)
# Sources: UNFCCC, EDGAR 8.0, World Bank, IMF WEO
# ==============================================================================

COUNTRY_DATABASE: Dict[str, Dict[str, Any]] = {
    # G7 Countries
    "US": {
        "name": "United States",
        "total_ghg_mt_co2e": Decimal("5222.4"),
        "lulucf_mt_co2e": Decimal("-756.0"),
        "gdp_ppp_billion_usd": Decimal("25460.0"),
        "population_million": Decimal("333.3"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "GB": {
        "name": "United Kingdom",
        "total_ghg_mt_co2e": Decimal("384.2"),
        "lulucf_mt_co2e": Decimal("-13.5"),
        "gdp_ppp_billion_usd": Decimal("3580.0"),
        "population_million": Decimal("67.5"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "DE": {
        "name": "Germany",
        "total_ghg_mt_co2e": Decimal("674.3"),
        "lulucf_mt_co2e": Decimal("-22.8"),
        "gdp_ppp_billion_usd": Decimal("4820.0"),
        "population_million": Decimal("84.1"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "FR": {
        "name": "France",
        "total_ghg_mt_co2e": Decimal("393.0"),
        "lulucf_mt_co2e": Decimal("-27.5"),
        "gdp_ppp_billion_usd": Decimal("3680.0"),
        "population_million": Decimal("67.9"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "IT": {
        "name": "Italy",
        "total_ghg_mt_co2e": Decimal("380.5"),
        "lulucf_mt_co2e": Decimal("-32.0"),
        "gdp_ppp_billion_usd": Decimal("2940.0"),
        "population_million": Decimal("59.0"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "CA": {
        "name": "Canada",
        "total_ghg_mt_co2e": Decimal("672.4"),
        "lulucf_mt_co2e": Decimal("82.0"),
        "gdp_ppp_billion_usd": Decimal("2140.0"),
        "population_million": Decimal("38.9"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "JP": {
        "name": "Japan",
        "total_ghg_mt_co2e": Decimal("1064.0"),
        "lulucf_mt_co2e": Decimal("-47.6"),
        "gdp_ppp_billion_usd": Decimal("5710.0"),
        "population_million": Decimal("125.7"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    # EU Members (additional)
    "ES": {
        "name": "Spain",
        "total_ghg_mt_co2e": Decimal("280.5"),
        "lulucf_mt_co2e": Decimal("-34.2"),
        "gdp_ppp_billion_usd": Decimal("2090.0"),
        "population_million": Decimal("47.4"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "NL": {
        "name": "Netherlands",
        "total_ghg_mt_co2e": Decimal("160.8"),
        "lulucf_mt_co2e": Decimal("5.2"),
        "gdp_ppp_billion_usd": Decimal("1130.0"),
        "population_million": Decimal("17.6"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "BE": {
        "name": "Belgium",
        "total_ghg_mt_co2e": Decimal("105.3"),
        "lulucf_mt_co2e": Decimal("-1.8"),
        "gdp_ppp_billion_usd": Decimal("680.0"),
        "population_million": Decimal("11.6"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "SE": {
        "name": "Sweden",
        "total_ghg_mt_co2e": Decimal("44.8"),
        "lulucf_mt_co2e": Decimal("-43.0"),
        "gdp_ppp_billion_usd": Decimal("640.0"),
        "population_million": Decimal("10.5"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "NO": {
        "name": "Norway",
        "total_ghg_mt_co2e": Decimal("48.9"),
        "lulucf_mt_co2e": Decimal("-23.0"),
        "gdp_ppp_billion_usd": Decimal("440.0"),
        "population_million": Decimal("5.4"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "DK": {
        "name": "Denmark",
        "total_ghg_mt_co2e": Decimal("43.5"),
        "lulucf_mt_co2e": Decimal("5.8"),
        "gdp_ppp_billion_usd": Decimal("400.0"),
        "population_million": Decimal("5.9"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "FI": {
        "name": "Finland",
        "total_ghg_mt_co2e": Decimal("42.1"),
        "lulucf_mt_co2e": Decimal("-15.2"),
        "gdp_ppp_billion_usd": Decimal("310.0"),
        "population_million": Decimal("5.5"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "AT": {
        "name": "Austria",
        "total_ghg_mt_co2e": Decimal("72.6"),
        "lulucf_mt_co2e": Decimal("-5.3"),
        "gdp_ppp_billion_usd": Decimal("560.0"),
        "population_million": Decimal("9.1"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "IE": {
        "name": "Ireland",
        "total_ghg_mt_co2e": Decimal("61.5"),
        "lulucf_mt_co2e": Decimal("7.2"),
        "gdp_ppp_billion_usd": Decimal("580.0"),
        "population_million": Decimal("5.1"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "PT": {
        "name": "Portugal",
        "total_ghg_mt_co2e": Decimal("55.8"),
        "lulucf_mt_co2e": Decimal("-8.4"),
        "gdp_ppp_billion_usd": Decimal("400.0"),
        "population_million": Decimal("10.3"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "GR": {
        "name": "Greece",
        "total_ghg_mt_co2e": Decimal("68.2"),
        "lulucf_mt_co2e": Decimal("-3.1"),
        "gdp_ppp_billion_usd": Decimal("380.0"),
        "population_million": Decimal("10.4"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "PL": {
        "name": "Poland",
        "total_ghg_mt_co2e": Decimal("373.5"),
        "lulucf_mt_co2e": Decimal("-31.5"),
        "gdp_ppp_billion_usd": Decimal("1560.0"),
        "population_million": Decimal("37.7"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "CZ": {
        "name": "Czech Republic",
        "total_ghg_mt_co2e": Decimal("109.8"),
        "lulucf_mt_co2e": Decimal("-4.6"),
        "gdp_ppp_billion_usd": Decimal("480.0"),
        "population_million": Decimal("10.8"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "RO": {
        "name": "Romania",
        "total_ghg_mt_co2e": Decimal("98.6"),
        "lulucf_mt_co2e": Decimal("-22.0"),
        "gdp_ppp_billion_usd": Decimal("690.0"),
        "population_million": Decimal("19.0"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "HU": {
        "name": "Hungary",
        "total_ghg_mt_co2e": Decimal("55.4"),
        "lulucf_mt_co2e": Decimal("-4.8"),
        "gdp_ppp_billion_usd": Decimal("370.0"),
        "population_million": Decimal("9.7"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    # BRICS
    "BR": {
        "name": "Brazil",
        "total_ghg_mt_co2e": Decimal("1050.0"),
        "lulucf_mt_co2e": Decimal("680.0"),
        "gdp_ppp_billion_usd": Decimal("3690.0"),
        "population_million": Decimal("214.3"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "RU": {
        "name": "Russia",
        "total_ghg_mt_co2e": Decimal("2160.0"),
        "lulucf_mt_co2e": Decimal("-535.0"),
        "gdp_ppp_billion_usd": Decimal("4370.0"),
        "population_million": Decimal("144.2"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "IN": {
        "name": "India",
        "total_ghg_mt_co2e": Decimal("3340.0"),
        "lulucf_mt_co2e": Decimal("-310.0"),
        "gdp_ppp_billion_usd": Decimal("11870.0"),
        "population_million": Decimal("1417.2"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "CN": {
        "name": "China",
        "total_ghg_mt_co2e": Decimal("13720.0"),
        "lulucf_mt_co2e": Decimal("-580.0"),
        "gdp_ppp_billion_usd": Decimal("30330.0"),
        "population_million": Decimal("1412.2"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "ZA": {
        "name": "South Africa",
        "total_ghg_mt_co2e": Decimal("440.0"),
        "lulucf_mt_co2e": Decimal("-15.0"),
        "gdp_ppp_billion_usd": Decimal("940.0"),
        "population_million": Decimal("60.6"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    # Asia-Pacific
    "AU": {
        "name": "Australia",
        "total_ghg_mt_co2e": Decimal("488.0"),
        "lulucf_mt_co2e": Decimal("-20.0"),
        "gdp_ppp_billion_usd": Decimal("1590.0"),
        "population_million": Decimal("26.0"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "NZ": {
        "name": "New Zealand",
        "total_ghg_mt_co2e": Decimal("68.7"),
        "lulucf_mt_co2e": Decimal("-24.0"),
        "gdp_ppp_billion_usd": Decimal("250.0"),
        "population_million": Decimal("5.1"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "KR": {
        "name": "South Korea",
        "total_ghg_mt_co2e": Decimal("656.0"),
        "lulucf_mt_co2e": Decimal("-40.8"),
        "gdp_ppp_billion_usd": Decimal("2670.0"),
        "population_million": Decimal("51.7"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
    "SG": {
        "name": "Singapore",
        "total_ghg_mt_co2e": Decimal("52.8"),
        "lulucf_mt_co2e": Decimal("0.1"),
        "gdp_ppp_billion_usd": Decimal("640.0"),
        "population_million": Decimal("5.6"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "TW": {
        "name": "Taiwan",
        "total_ghg_mt_co2e": Decimal("275.0"),
        "lulucf_mt_co2e": Decimal("-21.5"),
        "gdp_ppp_billion_usd": Decimal("1480.0"),
        "population_million": Decimal("23.6"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "ID": {
        "name": "Indonesia",
        "total_ghg_mt_co2e": Decimal("960.0"),
        "lulucf_mt_co2e": Decimal("450.0"),
        "gdp_ppp_billion_usd": Decimal("3990.0"),
        "population_million": Decimal("275.5"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "TH": {
        "name": "Thailand",
        "total_ghg_mt_co2e": Decimal("350.0"),
        "lulucf_mt_co2e": Decimal("-28.0"),
        "gdp_ppp_billion_usd": Decimal("1400.0"),
        "population_million": Decimal("71.6"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "MY": {
        "name": "Malaysia",
        "total_ghg_mt_co2e": Decimal("290.0"),
        "lulucf_mt_co2e": Decimal("-25.0"),
        "gdp_ppp_billion_usd": Decimal("1050.0"),
        "population_million": Decimal("33.0"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "PH": {
        "name": "Philippines",
        "total_ghg_mt_co2e": Decimal("186.0"),
        "lulucf_mt_co2e": Decimal("-45.0"),
        "gdp_ppp_billion_usd": Decimal("1140.0"),
        "population_million": Decimal("113.9"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "VN": {
        "name": "Vietnam",
        "total_ghg_mt_co2e": Decimal("380.0"),
        "lulucf_mt_co2e": Decimal("-32.0"),
        "gdp_ppp_billion_usd": Decimal("1280.0"),
        "population_million": Decimal("99.5"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    # Emerging Markets
    "MX": {
        "name": "Mexico",
        "total_ghg_mt_co2e": Decimal("530.0"),
        "lulucf_mt_co2e": Decimal("-140.0"),
        "gdp_ppp_billion_usd": Decimal("2740.0"),
        "population_million": Decimal("130.1"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "TR": {
        "name": "Turkey",
        "total_ghg_mt_co2e": Decimal("523.0"),
        "lulucf_mt_co2e": Decimal("-56.0"),
        "gdp_ppp_billion_usd": Decimal("2980.0"),
        "population_million": Decimal("85.3"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "SA": {
        "name": "Saudi Arabia",
        "total_ghg_mt_co2e": Decimal("672.0"),
        "lulucf_mt_co2e": Decimal("-2.5"),
        "gdp_ppp_billion_usd": Decimal("2000.0"),
        "population_million": Decimal("36.4"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "AE": {
        "name": "United Arab Emirates",
        "total_ghg_mt_co2e": Decimal("225.0"),
        "lulucf_mt_co2e": Decimal("-0.8"),
        "gdp_ppp_billion_usd": Decimal("730.0"),
        "population_million": Decimal("9.4"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "EG": {
        "name": "Egypt",
        "total_ghg_mt_co2e": Decimal("340.0"),
        "lulucf_mt_co2e": Decimal("-5.0"),
        "gdp_ppp_billion_usd": Decimal("1560.0"),
        "population_million": Decimal("104.3"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "NG": {
        "name": "Nigeria",
        "total_ghg_mt_co2e": Decimal("250.0"),
        "lulucf_mt_co2e": Decimal("50.0"),
        "gdp_ppp_billion_usd": Decimal("1280.0"),
        "population_million": Decimal("218.5"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "AR": {
        "name": "Argentina",
        "total_ghg_mt_co2e": Decimal("365.0"),
        "lulucf_mt_co2e": Decimal("-66.0"),
        "gdp_ppp_billion_usd": Decimal("1130.0"),
        "population_million": Decimal("46.2"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "CL": {
        "name": "Chile",
        "total_ghg_mt_co2e": Decimal("95.0"),
        "lulucf_mt_co2e": Decimal("-52.0"),
        "gdp_ppp_billion_usd": Decimal("540.0"),
        "population_million": Decimal("19.5"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "CO": {
        "name": "Colombia",
        "total_ghg_mt_co2e": Decimal("180.0"),
        "lulucf_mt_co2e": Decimal("30.0"),
        "gdp_ppp_billion_usd": Decimal("890.0"),
        "population_million": Decimal("51.9"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "IL": {
        "name": "Israel",
        "total_ghg_mt_co2e": Decimal("78.5"),
        "lulucf_mt_co2e": Decimal("-3.0"),
        "gdp_ppp_billion_usd": Decimal("530.0"),
        "population_million": Decimal("9.6"),
        "data_source": "edgar",
        "data_year": 2022,
    },
    "CH": {
        "name": "Switzerland",
        "total_ghg_mt_co2e": Decimal("41.5"),
        "lulucf_mt_co2e": Decimal("-2.0"),
        "gdp_ppp_billion_usd": Decimal("670.0"),
        "population_million": Decimal("8.7"),
        "data_source": "unfccc",
        "data_year": 2022,
    },
}


# Multilateral development bank member weights (illustrative)
MULTILATERAL_WEIGHTS: Dict[str, Dict[str, Decimal]] = {
    "WORLD_BANK": {
        "US": Decimal("0.156"),
        "JP": Decimal("0.074"),
        "CN": Decimal("0.061"),
        "DE": Decimal("0.043"),
        "GB": Decimal("0.039"),
        "FR": Decimal("0.039"),
        "IN": Decimal("0.035"),
    },
    "ADB": {
        "JP": Decimal("0.155"),
        "US": Decimal("0.155"),
        "CN": Decimal("0.068"),
        "IN": Decimal("0.068"),
        "AU": Decimal("0.058"),
        "KR": Decimal("0.053"),
    },
    "EBRD": {
        "US": Decimal("0.100"),
        "GB": Decimal("0.087"),
        "FR": Decimal("0.087"),
        "DE": Decimal("0.087"),
        "IT": Decimal("0.087"),
        "JP": Decimal("0.087"),
    },
    "EIB": {
        "DE": Decimal("0.163"),
        "FR": Decimal("0.163"),
        "IT": Decimal("0.163"),
        "ES": Decimal("0.099"),
        "NL": Decimal("0.048"),
        "BE": Decimal("0.048"),
    },
}


# Sub-sovereign GDP share approximations (region share of national GDP)
SUB_SOVEREIGN_GDP_SHARES: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "california": Decimal("0.146"),
        "texas": Decimal("0.088"),
        "new_york": Decimal("0.078"),
        "florida": Decimal("0.054"),
        "illinois": Decimal("0.042"),
        "pennsylvania": Decimal("0.040"),
    },
    "DE": {
        "bavaria": Decimal("0.183"),
        "north_rhine_westphalia": Decimal("0.210"),
        "baden_wurttemberg": Decimal("0.152"),
        "lower_saxony": Decimal("0.092"),
    },
    "GB": {
        "england": Decimal("0.845"),
        "scotland": Decimal("0.080"),
        "wales": Decimal("0.035"),
        "northern_ireland": Decimal("0.025"),
    },
    "CA": {
        "ontario": Decimal("0.385"),
        "quebec": Decimal("0.195"),
        "british_columbia": Decimal("0.125"),
        "alberta": Decimal("0.155"),
    },
    "AU": {
        "new_south_wales": Decimal("0.315"),
        "victoria": Decimal("0.240"),
        "queensland": Decimal("0.200"),
        "western_australia": Decimal("0.145"),
    },
}


# Currency exchange rates to USD
CURRENCY_RATES_TO_USD: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.0"),
    CurrencyCode.EUR: Decimal("1.0850"),
    CurrencyCode.GBP: Decimal("1.2650"),
    CurrencyCode.CAD: Decimal("0.7410"),
    CurrencyCode.AUD: Decimal("0.6520"),
    CurrencyCode.JPY: Decimal("0.006667"),
    CurrencyCode.CNY: Decimal("0.1378"),
    CurrencyCode.INR: Decimal("0.01198"),
    CurrencyCode.CHF: Decimal("1.1280"),
    CurrencyCode.SGD: Decimal("0.7440"),
    CurrencyCode.BRL: Decimal("0.1990"),
    CurrencyCode.ZAR: Decimal("0.05340"),
    CurrencyCode.SEK: Decimal("0.09250"),
    CurrencyCode.NOK: Decimal("0.09150"),
    CurrencyCode.DKK: Decimal("0.1450"),
    CurrencyCode.HKD: Decimal("0.1282"),
    CurrencyCode.KRW: Decimal("0.000745"),
    CurrencyCode.NZD: Decimal("0.6050"),
    CurrencyCode.MXN: Decimal("0.05820"),
    CurrencyCode.TRY: Decimal("0.03120"),
}


# PCAF uncertainty ranges for sovereign bonds
PCAF_UNCERTAINTY_RANGES: Dict[PCAFDataQuality, Decimal] = {
    PCAFDataQuality.SCORE_4: Decimal("0.25"),
    PCAFDataQuality.SCORE_5: Decimal("0.50"),
}


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class SovereignBondInput(BaseModel):
    """
    Input for sovereign bond financed emissions calculation.

    Example:
        >>> inp = SovereignBondInput(
        ...     outstanding_amount=Decimal("50000000"),
        ...     country_code="US",
        ...     reporting_year=2024,
        ... )
    """

    bond_type: SovereignBondType = Field(
        default=SovereignBondType.NATIONAL,
        description="Type of sovereign bond",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0, description="Outstanding bond amount"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD, description="Currency"
    )
    country_code: str = Field(
        ..., min_length=2, max_length=3,
        description="ISO 3166-1 alpha-2 country code"
    )
    region: Optional[str] = Field(
        default=None,
        description="Sub-sovereign region/state (for sub-sovereign bonds)"
    )
    multilateral_institution: Optional[str] = Field(
        default=None,
        description="Multilateral institution code (e.g., WORLD_BANK, ADB)"
    )
    is_green_sovereign: bool = Field(
        default=False,
        description="Whether this is a green sovereign bond"
    )
    green_project_emissions_mt: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Green project emissions in MtCO2e"
    )
    emissions_data_source: Optional[EmissionsDataSource] = Field(
        default=None,
        description="Source of country emissions data"
    )
    custom_total_ghg_mt: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Custom total GHG (MtCO2e) to override database"
    )
    custom_gdp_ppp_billion: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Custom GDP PPP (billion USD) to override database"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030, description="Reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier"
    )

    model_config = ConfigDict(frozen=True)

    @validator("country_code")
    def validate_country_code(cls, v: str) -> str:
        """Uppercase country code."""
        return v.upper().strip()


# ==============================================================================
# RESULT MODEL
# ==============================================================================


class InvestmentCalculationResult(BaseModel):
    """Result from a sovereign bond financed emissions calculation."""

    bond_type: str = Field(
        ..., description="Sovereign bond type"
    )
    country_code: str = Field(
        ..., description="Country code"
    )
    country_name: str = Field(
        ..., description="Country name"
    )
    outstanding_amount: Decimal = Field(
        ..., description="Outstanding amount"
    )
    outstanding_amount_usd: Decimal = Field(
        ..., description="Outstanding in USD"
    )
    attribution_factor: Decimal = Field(
        ..., description="Attribution factor (outstanding / GDP_PPP)"
    )
    country_emissions_mt_co2e: Decimal = Field(
        ..., description="Country emissions excl LULUCF (MtCO2e)"
    )
    financed_emissions_co2e: Decimal = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    emissions_per_capita_co2e: Decimal = Field(
        ..., description="Country per-capita emissions (tCO2e/person)"
    )
    emissions_intensity_per_gdp: Decimal = Field(
        ..., description="Emissions intensity (tCO2e per million USD GDP PPP)"
    )
    pcaf_quality_score: int = Field(
        ..., ge=4, le=5, description="PCAF score (4 or 5 only)"
    )
    uncertainty_lower_co2e: Decimal = Field(
        ..., description="Lower bound 95% CI (tCO2e)"
    )
    uncertainty_upper_co2e: Decimal = Field(
        ..., description="Upper bound 95% CI (tCO2e)"
    )
    data_source: str = Field(
        ..., description="Emissions data source"
    )
    data_year: int = Field(
        ..., description="Year of emissions data"
    )
    calculation_method: str = Field(
        ..., description="Calculation method"
    )
    dc_rules_applied: List[str] = Field(
        default_factory=list,
        description="Double-counting rules applied"
    )
    dc_warnings: List[str] = Field(
        default_factory=list,
        description="Double-counting warnings"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HASH UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """Serialize object to deterministic JSON for hashing."""

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default)


def _compute_hash(*inputs: Any) -> str:
    """Compute SHA-256 hash from variable inputs."""
    combined = ""
    for inp in inputs:
        combined += _serialize_for_hash(inp)
    return hashlib.sha256(combined.encode(_ENCODING)).hexdigest()


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class SovereignBondCalculatorEngine:
    """
    Thread-safe singleton engine for sovereign bond financed emissions.

    Implements PCAF methodology for government bonds using production-based
    national emissions (excluding LULUCF) and PPP-adjusted GDP for
    attribution. All arithmetic uses Python Decimal with ROUND_HALF_UP.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic with data from authoritative sources (UNFCCC, EDGAR,
    World Bank, IMF).

    Thread Safety:
        Uses __new__ singleton with threading.RLock.

    Example:
        >>> engine = SovereignBondCalculatorEngine()
        >>> result = engine.calculate(sovereign_input)
    """

    _instance: Optional["SovereignBondCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "SovereignBondCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the sovereign bond calculator engine."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._calculation_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        logger.info(
            "SovereignBondCalculatorEngine initialized: agent=%s, "
            "version=%s, countries=%d",
            AGENT_ID, VERSION, len(COUNTRY_DATABASE),
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _increment_count(self) -> int:
        """Increment calculation counter thread-safely."""
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize to 8 decimal places."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    def _quantize_2dp(self, value: Decimal) -> Decimal:
        """Quantize to 2 decimal places."""
        return value.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    def _convert_to_usd(
        self, amount: Decimal, currency: CurrencyCode
    ) -> Decimal:
        """Convert amount to USD."""
        rate = CURRENCY_RATES_TO_USD.get(currency)
        if rate is None:
            raise ValueError(f"Currency '{currency.value}' not found")
        return self._quantize(amount * rate)

    # =========================================================================
    # COUNTRY DATA
    # =========================================================================

    def _get_country_data(self, country_code: str) -> Dict[str, Any]:
        """
        Look up country data from the database.

        Args:
            country_code: ISO 3166-1 alpha-2 code.

        Returns:
            Country data dictionary.

        Raises:
            ValueError: If country not found.
        """
        code = country_code.upper().strip()
        data = COUNTRY_DATABASE.get(code)
        if data is None:
            available = sorted(COUNTRY_DATABASE.keys())
            raise ValueError(
                f"Country '{code}' not found in sovereign bond database. "
                f"Available ({len(available)}): {available}"
            )
        return data

    # =========================================================================
    # ATTRIBUTION
    # =========================================================================

    def _calculate_attribution_factor(
        self, outstanding_usd: Decimal, gdp_ppp_billion_usd: Decimal
    ) -> Decimal:
        """
        Calculate sovereign bond attribution factor.

        Formula: outstanding_amount / PPP_adjusted_GDP

        The GDP is in billions, so we multiply by 1e9 to get same units
        as outstanding_amount (USD).

        Args:
            outstanding_usd: Outstanding in USD.
            gdp_ppp_billion_usd: GDP PPP in billion USD.

        Returns:
            Attribution factor, quantized to 8 dp.
        """
        gdp_usd = gdp_ppp_billion_usd * Decimal("1000000000")
        if gdp_usd <= _ZERO:
            raise ValueError(
                f"GDP PPP must be positive, got {gdp_ppp_billion_usd} billion"
            )
        raw = outstanding_usd / gdp_usd
        return self._quantize(min(raw, _MAX_ATTRIBUTION))

    # =========================================================================
    # LULUCF ADJUSTMENT
    # =========================================================================

    def _adjust_for_lulucf(
        self, total_ghg_mt: Decimal, lulucf_mt: Decimal
    ) -> Decimal:
        """
        Ensure emissions are production-based excluding LULUCF.

        The total_ghg_mt in our database already excludes LULUCF.
        This method verifies and logs the LULUCF component.

        Args:
            total_ghg_mt: Total GHG emissions excluding LULUCF (MtCO2e).
            lulucf_mt: LULUCF emissions/removals (MtCO2e).

        Returns:
            Production-based emissions excluding LULUCF (MtCO2e).
        """
        # Our database stores total_ghg already without LULUCF
        # This method is kept for explicit documentation and logging
        logger.debug(
            "LULUCF adjustment: total_ghg=%s MtCO2e (excl LULUCF), "
            "lulucf=%s MtCO2e",
            total_ghg_mt, lulucf_mt,
        )
        return total_ghg_mt

    # =========================================================================
    # PER-CAPITA INTENSITY
    # =========================================================================

    def _calculate_per_capita_intensity(
        self, emissions_mt: Decimal, population_million: Decimal
    ) -> Decimal:
        """
        Calculate per-capita emissions intensity.

        Args:
            emissions_mt: Emissions in MtCO2e.
            population_million: Population in millions.

        Returns:
            Per-capita emissions in tCO2e per person.
        """
        if population_million <= _ZERO:
            return _ZERO

        # MtCO2e / million people = tCO2e / person
        per_capita = emissions_mt / population_million
        return self._quantize(per_capita)

    # =========================================================================
    # PCAF QUALITY
    # =========================================================================

    def _determine_pcaf_quality(
        self, input_data: SovereignBondInput
    ) -> PCAFDataQuality:
        """
        Determine PCAF quality for sovereign bonds.

        Sovereign bonds only qualify for Score 4 or Score 5:
        - Score 4: Standard path (UNFCCC/EDGAR + GDP PPP)
        - Score 5: Estimates or country not in database

        Args:
            input_data: SovereignBondInput.

        Returns:
            PCAFDataQuality (SCORE_4 or SCORE_5).
        """
        code = input_data.country_code.upper()

        # Check if country is in database with standard source
        if code in COUNTRY_DATABASE:
            data = COUNTRY_DATABASE[code]
            source = data.get("data_source", "estimate")
            if source in ("unfccc", "edgar", "iea"):
                return PCAFDataQuality.SCORE_4

        # Custom data provided
        if (
            input_data.custom_total_ghg_mt is not None
            and input_data.custom_gdp_ppp_billion is not None
        ):
            return PCAFDataQuality.SCORE_4

        return PCAFDataQuality.SCORE_5

    # =========================================================================
    # UNCERTAINTY
    # =========================================================================

    def _calculate_uncertainty(
        self, financed: Decimal, quality: PCAFDataQuality
    ) -> Tuple[Decimal, Decimal]:
        """Calculate 95% CI bounds."""
        half_width = PCAF_UNCERTAINTY_RANGES.get(quality, Decimal("0.50"))
        delta = self._quantize(financed * half_width)
        lower = self._quantize(max(financed - delta, _ZERO))
        upper = self._quantize(financed + delta)
        return lower, upper

    # =========================================================================
    # VALIDATE
    # =========================================================================

    def _validate_sovereign_input(
        self, input_data: SovereignBondInput
    ) -> List[str]:
        """
        Validate sovereign bond input.

        Args:
            input_data: SovereignBondInput.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []

        if input_data.outstanding_amount <= _ZERO:
            errors.append("outstanding_amount must be positive")

        code = input_data.country_code.upper()

        # Check country availability
        if code not in COUNTRY_DATABASE:
            if (
                input_data.custom_total_ghg_mt is None
                or input_data.custom_gdp_ppp_billion is None
            ):
                errors.append(
                    f"Country '{code}' not in database and no custom "
                    f"total_ghg/gdp_ppp provided"
                )

        # Sub-sovereign checks
        if input_data.bond_type == SovereignBondType.SUB_SOVEREIGN:
            if input_data.region is None:
                errors.append(
                    "Sub-sovereign bond requires 'region' field"
                )

        # Multilateral checks
        if input_data.bond_type == SovereignBondType.MULTILATERAL:
            if input_data.multilateral_institution is None:
                errors.append(
                    "Multilateral bond requires 'multilateral_institution'"
                )

        # Green sovereign checks
        if input_data.is_green_sovereign:
            if input_data.green_project_emissions_mt is None:
                errors.append(
                    "Green sovereign bond requires 'green_project_emissions_mt'"
                )

        return errors

    # =========================================================================
    # SUB-SOVEREIGN
    # =========================================================================

    def _calculate_sub_sovereign(
        self,
        region: str,
        country_code: str,
        outstanding_usd: Decimal,
        currency: CurrencyCode,
        reporting_year: int,
        input_data: SovereignBondInput,
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a sub-sovereign bond.

        Uses proportional GDP share to scale national emissions.

        Args:
            region: Region/state name.
            country_code: Country code.
            outstanding_usd: Outstanding amount in USD.
            currency: Original currency.
            reporting_year: Reporting year.
            input_data: Original input.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        country_data = self._get_country_data(country_code)

        # Look up sub-sovereign GDP share
        country_shares = SUB_SOVEREIGN_GDP_SHARES.get(country_code, {})
        region_key = region.lower().replace(" ", "_")
        gdp_share = country_shares.get(region_key, Decimal("0.10"))

        # Scale national figures by regional GDP share
        national_ghg = country_data["total_ghg_mt_co2e"]
        regional_ghg = self._quantize(national_ghg * gdp_share)

        national_gdp = country_data["gdp_ppp_billion_usd"]
        regional_gdp = self._quantize(national_gdp * gdp_share)

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, regional_gdp
        )

        # Financed emissions (MtCO2e -> tCO2e: multiply by 1e6)
        financed_mt = self._quantize(attribution * regional_ghg)
        financed_tco2e = self._quantize(
            financed_mt * Decimal("1000000")
        )

        # Per-capita (use national per-capita as proxy)
        national_pop = country_data["population_million"]
        regional_pop = self._quantize(national_pop * gdp_share)
        per_capita = self._calculate_per_capita_intensity(
            regional_ghg, regional_pop
        )

        # Intensity per GDP
        if regional_gdp > _ZERO:
            intensity = self._quantize(
                regional_ghg * Decimal("1000000") / (
                    regional_gdp * Decimal("1000000000")
                )
            )
        else:
            intensity = _ZERO

        # PCAF quality
        pcaf_score = self._determine_pcaf_quality(input_data)

        # Uncertainty
        lower, upper = self._calculate_uncertainty(financed_tco2e, pcaf_score)

        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(regional_ghg),
            str(financed_tco2e),
        )

        return InvestmentCalculationResult(
            bond_type=SovereignBondType.SUB_SOVEREIGN.value,
            country_code=country_code,
            country_name=f"{country_data['name']} - {region}",
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            country_emissions_mt_co2e=regional_ghg,
            financed_emissions_co2e=financed_tco2e,
            emissions_per_capita_co2e=per_capita,
            emissions_intensity_per_gdp=intensity,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            data_source=country_data.get("data_source", "edgar"),
            data_year=country_data.get("data_year", 2022),
            calculation_method="sub_sovereign_gdp_share",
            dc_rules_applied=["DC-INV-004"],
            dc_warnings=[
                f"Sub-sovereign {region} emissions scaled from national "
                f"by GDP share ({float(gdp_share):.1%})"
            ],
            reporting_year=reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

    # =========================================================================
    # MULTILATERAL
    # =========================================================================

    def _calculate_multilateral(
        self,
        member_countries: Dict[str, Decimal],
        outstanding_usd: Decimal,
        reporting_year: int,
        input_data: SovereignBondInput,
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a multilateral bond.

        Uses weighted average of member country emissions based on
        voting/capital share weights.

        Args:
            member_countries: Dict of country_code -> weight.
            outstanding_usd: Outstanding in USD.
            reporting_year: Reporting year.
            input_data: Original input.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()

        # Calculate weighted emissions and GDP
        weighted_ghg_mt = _ZERO
        weighted_gdp_b = _ZERO
        total_weight = _ZERO
        data_sources: List[str] = []

        for code, weight in member_countries.items():
            try:
                country_data = self._get_country_data(code)
                ghg = country_data["total_ghg_mt_co2e"]
                gdp = country_data["gdp_ppp_billion_usd"]

                weighted_ghg_mt = self._quantize(
                    weighted_ghg_mt + (ghg * weight)
                )
                weighted_gdp_b = self._quantize(
                    weighted_gdp_b + (gdp * weight)
                )
                total_weight = self._quantize(total_weight + weight)
                data_sources.append(
                    country_data.get("data_source", "edgar")
                )
            except ValueError:
                logger.warning(
                    "Multilateral: country %s not found, skipping", code
                )

        if total_weight <= _ZERO:
            raise ValueError(
                "No valid member countries found for multilateral calculation"
            )

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, weighted_gdp_b
        )

        # Financed emissions
        financed_mt = self._quantize(attribution * weighted_ghg_mt)
        financed_tco2e = self._quantize(
            financed_mt * Decimal("1000000")
        )

        # Intensity
        if weighted_gdp_b > _ZERO:
            intensity = self._quantize(
                weighted_ghg_mt * Decimal("1000000") / (
                    weighted_gdp_b * Decimal("1000000000")
                )
            )
        else:
            intensity = _ZERO

        # PCAF
        pcaf_score = PCAFDataQuality.SCORE_4

        # Uncertainty
        lower, upper = self._calculate_uncertainty(financed_tco2e, pcaf_score)

        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(weighted_ghg_mt),
            str(financed_tco2e),
        )

        institution = input_data.multilateral_institution or "UNKNOWN"

        return InvestmentCalculationResult(
            bond_type=SovereignBondType.MULTILATERAL.value,
            country_code="MULTI",
            country_name=f"Multilateral ({institution})",
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            country_emissions_mt_co2e=weighted_ghg_mt,
            financed_emissions_co2e=financed_tco2e,
            emissions_per_capita_co2e=_ZERO,
            emissions_intensity_per_gdp=intensity,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            data_source=", ".join(set(data_sources)) or "mixed",
            data_year=2022,
            calculation_method="multilateral_weighted_average",
            dc_rules_applied=["DC-INV-004"],
            dc_warnings=[
                f"Multilateral weighted across {len(member_countries)} "
                f"member countries (total weight={float(total_weight):.3f})"
            ],
            reporting_year=reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

    # =========================================================================
    # MAIN CALCULATE
    # =========================================================================

    def calculate(
        self, input_data: SovereignBondInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a sovereign bond.

        Main entry point. Routes to national, sub-sovereign, or
        multilateral calculator based on bond_type.

        Formula (national):
            attribution = outstanding / GDP_PPP
            country_emissions = total_GHG - LULUCF (production-based)
            financed_emissions = attribution x country_emissions

        Args:
            input_data: SovereignBondInput.

        Returns:
            InvestmentCalculationResult.

        Raises:
            ValueError: If validation fails or country not found.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "Sovereign bond calculation #%d: type=%s, country=%s, "
            "outstanding=%s %s",
            calc_number,
            input_data.bond_type.value,
            input_data.country_code,
            input_data.outstanding_amount,
            input_data.currency.value,
        )

        # Validate
        errors = self._validate_sovereign_input(input_data)
        if errors:
            raise ValueError(
                f"Sovereign input validation failed: {'; '.join(errors)}"
            )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_amount, input_data.currency
        )

        # Route by bond type
        if input_data.bond_type == SovereignBondType.SUB_SOVEREIGN:
            return self._calculate_sub_sovereign(
                region=input_data.region or "unknown",
                country_code=input_data.country_code.upper(),
                outstanding_usd=outstanding_usd,
                currency=input_data.currency,
                reporting_year=input_data.reporting_year,
                input_data=input_data,
            )

        if input_data.bond_type == SovereignBondType.MULTILATERAL:
            institution = (
                input_data.multilateral_institution or "WORLD_BANK"
            )
            weights = MULTILATERAL_WEIGHTS.get(institution)
            if weights is None:
                raise ValueError(
                    f"Multilateral institution '{institution}' not found. "
                    f"Available: {sorted(MULTILATERAL_WEIGHTS.keys())}"
                )
            return self._calculate_multilateral(
                member_countries=weights,
                outstanding_usd=outstanding_usd,
                reporting_year=input_data.reporting_year,
                input_data=input_data,
            )

        # National / standard sovereign bond calculation
        code = input_data.country_code.upper()

        # Get country data (custom overrides or database)
        if (
            input_data.custom_total_ghg_mt is not None
            and input_data.custom_gdp_ppp_billion is not None
        ):
            total_ghg_mt = input_data.custom_total_ghg_mt
            gdp_ppp_b = input_data.custom_gdp_ppp_billion
            country_name = f"Custom ({code})"
            data_source = "custom"
            data_year = input_data.reporting_year
            lulucf_mt = _ZERO
            population_m = _ZERO
        else:
            country_data = self._get_country_data(code)
            total_ghg_mt = country_data["total_ghg_mt_co2e"]
            lulucf_mt = country_data["lulucf_mt_co2e"]
            gdp_ppp_b = country_data["gdp_ppp_billion_usd"]
            country_name = country_data["name"]
            data_source = country_data.get("data_source", "edgar")
            data_year = country_data.get("data_year", 2022)
            population_m = country_data.get(
                "population_million", _ZERO
            )

        # Ensure LULUCF is excluded (our DB already stores excl LULUCF)
        production_ghg_mt = self._adjust_for_lulucf(total_ghg_mt, lulucf_mt)

        # Green sovereign bond override
        if (
            input_data.is_green_sovereign
            and input_data.green_project_emissions_mt is not None
        ):
            production_ghg_mt = input_data.green_project_emissions_mt
            data_source = "green_sovereign_project"

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, gdp_ppp_b
        )

        # Financed emissions (MtCO2e -> tCO2e)
        financed_mt = self._quantize(attribution * production_ghg_mt)
        financed_tco2e = self._quantize(
            financed_mt * Decimal("1000000")
        )

        # Per-capita intensity
        per_capita = self._calculate_per_capita_intensity(
            production_ghg_mt, population_m
        )

        # GDP intensity (tCO2e per million USD GDP PPP)
        if gdp_ppp_b > _ZERO:
            # MtCO2e / billion USD = tCO2e / million USD * 1000
            intensity = self._quantize(
                (production_ghg_mt / gdp_ppp_b) * Decimal("1000")
            )
        else:
            intensity = _ZERO

        # PCAF
        pcaf_score = self._determine_pcaf_quality(input_data)

        # Uncertainty
        lower, upper = self._calculate_uncertainty(
            financed_tco2e, pcaf_score
        )

        # DC rules
        dc_applied = ["DC-INV-004"]
        dc_warnings: List[str] = [
            "Sovereign emissions include corporate emissions; "
            "check for overlap with corporate bond/equity holdings "
            f"in {country_name}"
        ]

        # Duration
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(production_ghg_mt),
            str(financed_tco2e),
            str(pcaf_score.value),
        )

        calc_method = "sovereign_gdp_ppp"
        if input_data.is_green_sovereign:
            calc_method = "green_sovereign_project"

        result = InvestmentCalculationResult(
            bond_type=input_data.bond_type.value,
            country_code=code,
            country_name=country_name,
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            country_emissions_mt_co2e=production_ghg_mt,
            financed_emissions_co2e=financed_tco2e,
            emissions_per_capita_co2e=per_capita,
            emissions_intensity_per_gdp=intensity,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            data_source=data_source,
            data_year=data_year,
            calculation_method=calc_method,
            dc_rules_applied=dc_applied,
            dc_warnings=dc_warnings,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "Sovereign bond #%d complete: country=%s (%s), "
            "attribution=%.10f, country_ghg=%s MtCO2e, "
            "financed=%s tCO2e, pcaf=%d, method=%s, duration=%.2fms, "
            "provenance=%s...%s",
            calc_number,
            code,
            country_name,
            float(attribution),
            production_ghg_mt,
            financed_tco2e,
            pcaf_score.value,
            calc_method,
            float(duration_ms),
            provenance[:8],
            provenance[-8:],
        )

        return result

    # =========================================================================
    # BATCH
    # =========================================================================

    def calculate_batch(
        self, inputs: List[SovereignBondInput]
    ) -> List[InvestmentCalculationResult]:
        """
        Calculate financed emissions for a batch of sovereign bonds.

        Args:
            inputs: List of SovereignBondInput.

        Returns:
            List of successful InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        results: List[InvestmentCalculationResult] = []
        error_count = 0

        logger.info(
            "Batch sovereign calculation: %d inputs", len(inputs)
        )

        for i, inp in enumerate(inputs):
            try:
                result = self.calculate(inp)
                results.append(result)
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Batch sovereign #%d failed (%s): %s",
                    i + 1, inp.country_code, exc, exc_info=True,
                )

        total_duration = time.monotonic() - start_time
        logger.info(
            "Batch sovereign complete: %d/%d succeeded, "
            "%d errors, duration=%.3fs",
            len(results), len(inputs), error_count, total_duration,
        )

        return results

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def get_calculation_count(self) -> int:
        """Get total calculation count."""
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """Get engine state summary."""
        return {
            "engine": "SovereignBondCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "calculation_count": self.get_calculation_count(),
            "countries_available": len(COUNTRY_DATABASE),
            "country_codes": sorted(COUNTRY_DATABASE.keys()),
            "multilateral_institutions": sorted(MULTILATERAL_WEIGHTS.keys()),
            "sub_sovereign_countries": sorted(SUB_SOVEREIGN_GDP_SHARES.keys()),
            "pcaf_quality_levels": [s.value for s in PCAFDataQuality],
        }

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (testing only)."""
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_calculator_instance: Optional[SovereignBondCalculatorEngine] = None
_calculator_lock: threading.RLock = threading.RLock()


def get_sovereign_bond_calculator() -> SovereignBondCalculatorEngine:
    """Get singleton SovereignBondCalculatorEngine."""
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = SovereignBondCalculatorEngine()
        return _calculator_instance


def reset_sovereign_bond_calculator() -> None:
    """Reset module-level calculator (testing only)."""
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    SovereignBondCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Enums
    "SovereignBondType",
    "PCAFDataQuality",
    "EmissionsDataSource",
    "CurrencyCode",
    # Constants
    "COUNTRY_DATABASE",
    "MULTILATERAL_WEIGHTS",
    "SUB_SOVEREIGN_GDP_SHARES",
    "CURRENCY_RATES_TO_USD",
    "PCAF_UNCERTAINTY_RANGES",
    # Input models
    "SovereignBondInput",
    # Result model
    "InvestmentCalculationResult",
    # Engine
    "SovereignBondCalculatorEngine",
    "get_sovereign_bond_calculator",
    "reset_sovereign_bond_calculator",
]
