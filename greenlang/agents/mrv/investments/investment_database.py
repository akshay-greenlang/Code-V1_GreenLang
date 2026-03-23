# -*- coding: utf-8 -*-
"""
InvestmentDatabaseEngine - Reference data engine for investment emission factors.

This module implements the InvestmentDatabaseEngine for AGENT-MRV-028
(Investments, GHG Protocol Scope 3 Category 15). It provides thread-safe
singleton access to PCAF attribution rules, sector emission factors,
country-level emissions data, grid emission factors, building benchmarks,
vehicle emission factors, EEIO sector factors, PCAF data quality matrices,
currency conversion rates, sovereign country data, carbon intensity
benchmarks, double-counting rules, compliance framework rules, DQI scoring
criteria, uncertainty ranges, and portfolio alignment thresholds.

Features:
- 8 PCAF asset class attribution rules (PCAF Global Standard 2022)
- 12 GICS sector emission factors (tCO2e/$M revenue)
- 50+ country emission factors (total GHG, GDP PPP, per capita)
- 12 country + 26 eGRID subregion grid emission factors
- 6 property type x 5 climate zone building EUI benchmarks
- 5 vehicle category emission factors
- 12 EEIO sector factors (kgCO2e/$)
- 8 asset class x 5 score PCAF data quality matrix
- 15 currency conversion rates to USD
- 50 sovereign country data records (GDP PPP, emissions)
- Carbon intensity benchmarks by sector
- Double-counting prevention rules
- 7 compliance framework rule sets
- DQI scoring criteria, uncertainty ranges, portfolio alignment thresholds
- Thread-safe singleton pattern with __new__
- Zero-hallucination factor retrieval
- Provenance tracking via SHA-256 hashes
- Prometheus metrics recording for all lookups

Example:
    >>> engine = InvestmentDatabaseEngine()
    >>> rule = engine.get_pcaf_attribution_rule("listed_equity")
    >>> rule["formula"]
    'outstanding / EVIC'
    >>> ef = engine.get_sector_ef("energy")
    >>> ef
    Decimal('450')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import logging
import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")

# =============================================================================
# AGENT METADATA
# =============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"


# =============================================================================
# TABLE 1: PCAF ATTRIBUTION RULES
# =============================================================================

PCAF_ATTRIBUTION_RULES: Dict[str, Dict[str, Any]] = {
    "listed_equity": {
        "asset_class": "listed_equity",
        "description": "Publicly traded equity shares",
        "formula": "outstanding / EVIC",
        "numerator": "outstanding_amount",
        "denominator": "EVIC (market_cap + total_debt)",
        "pcaf_standard_ref": "Part A, Chapter 5.1",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": True,
        "data_requirements": [
            "outstanding_amount",
            "market_cap",
            "total_debt",
            "company_scope1_emissions",
            "company_scope2_emissions",
        ],
    },
    "corporate_bond": {
        "asset_class": "corporate_bond",
        "description": "Corporate bonds and fixed income instruments",
        "formula": "outstanding / EVIC",
        "numerator": "outstanding_amount",
        "denominator": "EVIC (market_cap + total_debt)",
        "pcaf_standard_ref": "Part A, Chapter 5.2",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": True,
        "data_requirements": [
            "outstanding_amount",
            "market_cap",
            "total_debt",
            "company_scope1_emissions",
            "company_scope2_emissions",
        ],
    },
    "private_equity": {
        "asset_class": "private_equity",
        "description": "Unlisted equity and private equity fund holdings",
        "formula": "outstanding / (total_equity + total_debt)",
        "numerator": "outstanding_amount",
        "denominator": "total_equity + total_debt",
        "pcaf_standard_ref": "Part A, Chapter 5.3",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": True,
        "data_requirements": [
            "outstanding_amount",
            "total_equity",
            "total_debt",
            "company_scope1_emissions",
            "company_scope2_emissions",
        ],
    },
    "project_finance": {
        "asset_class": "project_finance",
        "description": "Project finance loans for specific projects",
        "formula": "outstanding / total_project_cost",
        "numerator": "outstanding_amount",
        "denominator": "total_project_cost",
        "pcaf_standard_ref": "Part A, Chapter 5.4",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": False,
        "data_requirements": [
            "outstanding_amount",
            "total_project_cost",
            "project_scope1_emissions",
            "project_scope2_emissions",
        ],
    },
    "commercial_real_estate": {
        "asset_class": "commercial_real_estate",
        "description": "Commercial real estate loans and investments",
        "formula": "outstanding / property_value",
        "numerator": "outstanding_amount",
        "denominator": "property_value_at_origination",
        "pcaf_standard_ref": "Part A, Chapter 5.5",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": False,
        "data_requirements": [
            "outstanding_amount",
            "property_value",
            "property_type",
            "floor_area_sqm",
            "energy_consumption_kwh",
        ],
    },
    "mortgage": {
        "asset_class": "mortgage",
        "description": "Residential mortgage loans",
        "formula": "outstanding_loan / property_value",
        "numerator": "outstanding_loan_amount",
        "denominator": "property_value_at_origination",
        "pcaf_standard_ref": "Part A, Chapter 5.6",
        "scope_coverage": ["scope_1", "scope_2"],
        "optional_scope3": False,
        "data_requirements": [
            "outstanding_loan",
            "property_value",
            "building_type",
            "floor_area_sqm",
            "energy_label",
        ],
    },
    "motor_vehicle_loan": {
        "asset_class": "motor_vehicle_loan",
        "description": "Motor vehicle financing and leasing",
        "formula": "outstanding_loan / vehicle_value",
        "numerator": "outstanding_loan_amount",
        "denominator": "vehicle_value_at_origination",
        "pcaf_standard_ref": "Part A, Chapter 5.7",
        "scope_coverage": ["scope_1"],
        "optional_scope3": False,
        "data_requirements": [
            "outstanding_loan",
            "vehicle_value",
            "vehicle_type",
            "fuel_type",
            "annual_km",
        ],
    },
    "sovereign_bond": {
        "asset_class": "sovereign_bond",
        "description": "Sovereign and sub-sovereign debt",
        "formula": "outstanding / PPP_adjusted_GDP",
        "numerator": "outstanding_amount",
        "denominator": "PPP_adjusted_GDP",
        "pcaf_standard_ref": "Part A, Chapter 5.8",
        "scope_coverage": ["production_emissions"],
        "optional_scope3": False,
        "data_requirements": [
            "outstanding_amount",
            "country_code",
            "gdp_ppp",
            "country_total_ghg",
        ],
    },
}


# =============================================================================
# TABLE 2: SECTOR EMISSION FACTORS
# =============================================================================

SECTOR_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "gics_sector": "Energy",
        "gics_code": "10",
        "ef_tco2e_per_m_revenue": Decimal("450"),
        "description": "Oil, gas, and consumable fuels; energy equipment and services",
        "typical_scope1_pct": Decimal("0.75"),
        "typical_scope2_pct": Decimal("0.25"),
    },
    "materials": {
        "gics_sector": "Materials",
        "gics_code": "15",
        "ef_tco2e_per_m_revenue": Decimal("320"),
        "description": "Chemicals, construction materials, metals, mining, paper",
        "typical_scope1_pct": Decimal("0.65"),
        "typical_scope2_pct": Decimal("0.35"),
    },
    "industrials": {
        "gics_sector": "Industrials",
        "gics_code": "20",
        "ef_tco2e_per_m_revenue": Decimal("180"),
        "description": "Capital goods, commercial services, transportation",
        "typical_scope1_pct": Decimal("0.55"),
        "typical_scope2_pct": Decimal("0.45"),
    },
    "consumer_discretionary": {
        "gics_sector": "Consumer Discretionary",
        "gics_code": "25",
        "ef_tco2e_per_m_revenue": Decimal("85"),
        "description": "Automobiles, consumer durables, retailing, apparel",
        "typical_scope1_pct": Decimal("0.40"),
        "typical_scope2_pct": Decimal("0.60"),
    },
    "consumer_staples": {
        "gics_sector": "Consumer Staples",
        "gics_code": "30",
        "ef_tco2e_per_m_revenue": Decimal("95"),
        "description": "Food, beverages, tobacco, household products",
        "typical_scope1_pct": Decimal("0.45"),
        "typical_scope2_pct": Decimal("0.55"),
    },
    "healthcare": {
        "gics_sector": "Health Care",
        "gics_code": "35",
        "ef_tco2e_per_m_revenue": Decimal("45"),
        "description": "Pharmaceuticals, biotech, healthcare equipment, providers",
        "typical_scope1_pct": Decimal("0.30"),
        "typical_scope2_pct": Decimal("0.70"),
    },
    "financials": {
        "gics_sector": "Financials",
        "gics_code": "40",
        "ef_tco2e_per_m_revenue": Decimal("12"),
        "description": "Banks, insurance, diversified financials, fintech",
        "typical_scope1_pct": Decimal("0.15"),
        "typical_scope2_pct": Decimal("0.85"),
    },
    "information_technology": {
        "gics_sector": "Information Technology",
        "gics_code": "45",
        "ef_tco2e_per_m_revenue": Decimal("32"),
        "description": "Software, hardware, semiconductors, IT services",
        "typical_scope1_pct": Decimal("0.20"),
        "typical_scope2_pct": Decimal("0.80"),
    },
    "communication_services": {
        "gics_sector": "Communication Services",
        "gics_code": "50",
        "ef_tco2e_per_m_revenue": Decimal("28"),
        "description": "Telecom, media, entertainment, interactive media",
        "typical_scope1_pct": Decimal("0.20"),
        "typical_scope2_pct": Decimal("0.80"),
    },
    "utilities": {
        "gics_sector": "Utilities",
        "gics_code": "55",
        "ef_tco2e_per_m_revenue": Decimal("680"),
        "description": "Electric, gas, water utilities; independent power producers",
        "typical_scope1_pct": Decimal("0.85"),
        "typical_scope2_pct": Decimal("0.15"),
    },
    "real_estate": {
        "gics_sector": "Real Estate",
        "gics_code": "60",
        "ef_tco2e_per_m_revenue": Decimal("120"),
        "description": "REITs, real estate management, development",
        "typical_scope1_pct": Decimal("0.25"),
        "typical_scope2_pct": Decimal("0.75"),
    },
    "other": {
        "gics_sector": "Other / Unclassified",
        "gics_code": "99",
        "ef_tco2e_per_m_revenue": Decimal("150"),
        "description": "Unclassified or diversified sector",
        "typical_scope1_pct": Decimal("0.50"),
        "typical_scope2_pct": Decimal("0.50"),
    },
}


# =============================================================================
# TABLE 3: COUNTRY EMISSION FACTORS
# =============================================================================

COUNTRY_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "country_name": "United States",
        "total_ghg_mt": Decimal("5222"),
        "gdp_ppp_b": Decimal("25460"),
        "per_capita_tco2e": Decimal("15.52"),
        "iso_alpha2": "US",
        "iso_alpha3": "USA",
        "region": "North America",
    },
    "CN": {
        "country_name": "China",
        "total_ghg_mt": Decimal("12668"),
        "gdp_ppp_b": Decimal("30180"),
        "per_capita_tco2e": Decimal("8.99"),
        "iso_alpha2": "CN",
        "iso_alpha3": "CHN",
        "region": "East Asia",
    },
    "EU27": {
        "country_name": "European Union (27)",
        "total_ghg_mt": Decimal("3300"),
        "gdp_ppp_b": Decimal("20000"),
        "per_capita_tco2e": Decimal("7.38"),
        "iso_alpha2": "EU27",
        "iso_alpha3": "EU27",
        "region": "Europe",
    },
    "IN": {
        "country_name": "India",
        "total_ghg_mt": Decimal("3400"),
        "gdp_ppp_b": Decimal("13030"),
        "per_capita_tco2e": Decimal("2.39"),
        "iso_alpha2": "IN",
        "iso_alpha3": "IND",
        "region": "South Asia",
    },
    "JP": {
        "country_name": "Japan",
        "total_ghg_mt": Decimal("1064"),
        "gdp_ppp_b": Decimal("5700"),
        "per_capita_tco2e": Decimal("8.50"),
        "iso_alpha2": "JP",
        "iso_alpha3": "JPN",
        "region": "East Asia",
    },
    "GB": {
        "country_name": "United Kingdom",
        "total_ghg_mt": Decimal("384"),
        "gdp_ppp_b": Decimal("3630"),
        "per_capita_tco2e": Decimal("5.65"),
        "iso_alpha2": "GB",
        "iso_alpha3": "GBR",
        "region": "Europe",
    },
    "DE": {
        "country_name": "Germany",
        "total_ghg_mt": Decimal("674"),
        "gdp_ppp_b": Decimal("4830"),
        "per_capita_tco2e": Decimal("8.10"),
        "iso_alpha2": "DE",
        "iso_alpha3": "DEU",
        "region": "Europe",
    },
    "FR": {
        "country_name": "France",
        "total_ghg_mt": Decimal("305"),
        "gdp_ppp_b": Decimal("3560"),
        "per_capita_tco2e": Decimal("4.50"),
        "iso_alpha2": "FR",
        "iso_alpha3": "FRA",
        "region": "Europe",
    },
    "CA": {
        "country_name": "Canada",
        "total_ghg_mt": Decimal("672"),
        "gdp_ppp_b": Decimal("2230"),
        "per_capita_tco2e": Decimal("17.30"),
        "iso_alpha2": "CA",
        "iso_alpha3": "CAN",
        "region": "North America",
    },
    "AU": {
        "country_name": "Australia",
        "total_ghg_mt": Decimal("488"),
        "gdp_ppp_b": Decimal("1550"),
        "per_capita_tco2e": Decimal("18.80"),
        "iso_alpha2": "AU",
        "iso_alpha3": "AUS",
        "region": "Oceania",
    },
    "BR": {
        "country_name": "Brazil",
        "total_ghg_mt": Decimal("1280"),
        "gdp_ppp_b": Decimal("3680"),
        "per_capita_tco2e": Decimal("5.98"),
        "iso_alpha2": "BR",
        "iso_alpha3": "BRA",
        "region": "South America",
    },
    "RU": {
        "country_name": "Russia",
        "total_ghg_mt": Decimal("2150"),
        "gdp_ppp_b": Decimal("4490"),
        "per_capita_tco2e": Decimal("14.88"),
        "iso_alpha2": "RU",
        "iso_alpha3": "RUS",
        "region": "Eurasia",
    },
    "KR": {
        "country_name": "South Korea",
        "total_ghg_mt": Decimal("616"),
        "gdp_ppp_b": Decimal("2680"),
        "per_capita_tco2e": Decimal("11.89"),
        "iso_alpha2": "KR",
        "iso_alpha3": "KOR",
        "region": "East Asia",
    },
    "MX": {
        "country_name": "Mexico",
        "total_ghg_mt": Decimal("520"),
        "gdp_ppp_b": Decimal("2890"),
        "per_capita_tco2e": Decimal("3.98"),
        "iso_alpha2": "MX",
        "iso_alpha3": "MEX",
        "region": "North America",
    },
    "ID": {
        "country_name": "Indonesia",
        "total_ghg_mt": Decimal("960"),
        "gdp_ppp_b": Decimal("3860"),
        "per_capita_tco2e": Decimal("3.49"),
        "iso_alpha2": "ID",
        "iso_alpha3": "IDN",
        "region": "Southeast Asia",
    },
    "SA": {
        "country_name": "Saudi Arabia",
        "total_ghg_mt": Decimal("672"),
        "gdp_ppp_b": Decimal("1930"),
        "per_capita_tco2e": Decimal("18.65"),
        "iso_alpha2": "SA",
        "iso_alpha3": "SAU",
        "region": "Middle East",
    },
    "TR": {
        "country_name": "Turkey",
        "total_ghg_mt": Decimal("445"),
        "gdp_ppp_b": Decimal("3120"),
        "per_capita_tco2e": Decimal("5.21"),
        "iso_alpha2": "TR",
        "iso_alpha3": "TUR",
        "region": "Europe / Middle East",
    },
    "ZA": {
        "country_name": "South Africa",
        "total_ghg_mt": Decimal("480"),
        "gdp_ppp_b": Decimal("950"),
        "per_capita_tco2e": Decimal("7.91"),
        "iso_alpha2": "ZA",
        "iso_alpha3": "ZAF",
        "region": "Africa",
    },
    "IT": {
        "country_name": "Italy",
        "total_ghg_mt": Decimal("340"),
        "gdp_ppp_b": Decimal("3010"),
        "per_capita_tco2e": Decimal("5.76"),
        "iso_alpha2": "IT",
        "iso_alpha3": "ITA",
        "region": "Europe",
    },
    "ES": {
        "country_name": "Spain",
        "total_ghg_mt": Decimal("256"),
        "gdp_ppp_b": Decimal("2190"),
        "per_capita_tco2e": Decimal("5.40"),
        "iso_alpha2": "ES",
        "iso_alpha3": "ESP",
        "region": "Europe",
    },
    "PL": {
        "country_name": "Poland",
        "total_ghg_mt": Decimal("340"),
        "gdp_ppp_b": Decimal("1590"),
        "per_capita_tco2e": Decimal("8.98"),
        "iso_alpha2": "PL",
        "iso_alpha3": "POL",
        "region": "Europe",
    },
    "NL": {
        "country_name": "Netherlands",
        "total_ghg_mt": Decimal("156"),
        "gdp_ppp_b": Decimal("1160"),
        "per_capita_tco2e": Decimal("8.86"),
        "iso_alpha2": "NL",
        "iso_alpha3": "NLD",
        "region": "Europe",
    },
    "SE": {
        "country_name": "Sweden",
        "total_ghg_mt": Decimal("44"),
        "gdp_ppp_b": Decimal("650"),
        "per_capita_tco2e": Decimal("4.22"),
        "iso_alpha2": "SE",
        "iso_alpha3": "SWE",
        "region": "Europe",
    },
    "NO": {
        "country_name": "Norway",
        "total_ghg_mt": Decimal("49"),
        "gdp_ppp_b": Decimal("440"),
        "per_capita_tco2e": Decimal("9.00"),
        "iso_alpha2": "NO",
        "iso_alpha3": "NOR",
        "region": "Europe",
    },
    "CH": {
        "country_name": "Switzerland",
        "total_ghg_mt": Decimal("40"),
        "gdp_ppp_b": Decimal("700"),
        "per_capita_tco2e": Decimal("4.55"),
        "iso_alpha2": "CH",
        "iso_alpha3": "CHE",
        "region": "Europe",
    },
    "AT": {
        "country_name": "Austria",
        "total_ghg_mt": Decimal("72"),
        "gdp_ppp_b": Decimal("570"),
        "per_capita_tco2e": Decimal("7.96"),
        "iso_alpha2": "AT",
        "iso_alpha3": "AUT",
        "region": "Europe",
    },
    "BE": {
        "country_name": "Belgium",
        "total_ghg_mt": Decimal("104"),
        "gdp_ppp_b": Decimal("660"),
        "per_capita_tco2e": Decimal("8.92"),
        "iso_alpha2": "BE",
        "iso_alpha3": "BEL",
        "region": "Europe",
    },
    "DK": {
        "country_name": "Denmark",
        "total_ghg_mt": Decimal("32"),
        "gdp_ppp_b": Decimal("390"),
        "per_capita_tco2e": Decimal("5.44"),
        "iso_alpha2": "DK",
        "iso_alpha3": "DNK",
        "region": "Europe",
    },
    "FI": {
        "country_name": "Finland",
        "total_ghg_mt": Decimal("42"),
        "gdp_ppp_b": Decimal("310"),
        "per_capita_tco2e": Decimal("7.57"),
        "iso_alpha2": "FI",
        "iso_alpha3": "FIN",
        "region": "Europe",
    },
    "IE": {
        "country_name": "Ireland",
        "total_ghg_mt": Decimal("60"),
        "gdp_ppp_b": Decimal("610"),
        "per_capita_tco2e": Decimal("11.79"),
        "iso_alpha2": "IE",
        "iso_alpha3": "IRL",
        "region": "Europe",
    },
    "PT": {
        "country_name": "Portugal",
        "total_ghg_mt": Decimal("48"),
        "gdp_ppp_b": Decimal("410"),
        "per_capita_tco2e": Decimal("4.67"),
        "iso_alpha2": "PT",
        "iso_alpha3": "PRT",
        "region": "Europe",
    },
    "GR": {
        "country_name": "Greece",
        "total_ghg_mt": Decimal("62"),
        "gdp_ppp_b": Decimal("370"),
        "per_capita_tco2e": Decimal("5.87"),
        "iso_alpha2": "GR",
        "iso_alpha3": "GRC",
        "region": "Europe",
    },
    "CZ": {
        "country_name": "Czech Republic",
        "total_ghg_mt": Decimal("100"),
        "gdp_ppp_b": Decimal("490"),
        "per_capita_tco2e": Decimal("9.36"),
        "iso_alpha2": "CZ",
        "iso_alpha3": "CZE",
        "region": "Europe",
    },
    "RO": {
        "country_name": "Romania",
        "total_ghg_mt": Decimal("82"),
        "gdp_ppp_b": Decimal("680"),
        "per_capita_tco2e": Decimal("4.29"),
        "iso_alpha2": "RO",
        "iso_alpha3": "ROU",
        "region": "Europe",
    },
    "HU": {
        "country_name": "Hungary",
        "total_ghg_mt": Decimal("52"),
        "gdp_ppp_b": Decimal("370"),
        "per_capita_tco2e": Decimal("5.38"),
        "iso_alpha2": "HU",
        "iso_alpha3": "HUN",
        "region": "Europe",
    },
    "TH": {
        "country_name": "Thailand",
        "total_ghg_mt": Decimal("340"),
        "gdp_ppp_b": Decimal("1410"),
        "per_capita_tco2e": Decimal("4.86"),
        "iso_alpha2": "TH",
        "iso_alpha3": "THA",
        "region": "Southeast Asia",
    },
    "MY": {
        "country_name": "Malaysia",
        "total_ghg_mt": Decimal("290"),
        "gdp_ppp_b": Decimal("1060"),
        "per_capita_tco2e": Decimal("8.84"),
        "iso_alpha2": "MY",
        "iso_alpha3": "MYS",
        "region": "Southeast Asia",
    },
    "SG": {
        "country_name": "Singapore",
        "total_ghg_mt": Decimal("52"),
        "gdp_ppp_b": Decimal("680"),
        "per_capita_tco2e": Decimal("9.14"),
        "iso_alpha2": "SG",
        "iso_alpha3": "SGP",
        "region": "Southeast Asia",
    },
    "PH": {
        "country_name": "Philippines",
        "total_ghg_mt": Decimal("186"),
        "gdp_ppp_b": Decimal("1170"),
        "per_capita_tco2e": Decimal("1.66"),
        "iso_alpha2": "PH",
        "iso_alpha3": "PHL",
        "region": "Southeast Asia",
    },
    "VN": {
        "country_name": "Vietnam",
        "total_ghg_mt": Decimal("350"),
        "gdp_ppp_b": Decimal("1280"),
        "per_capita_tco2e": Decimal("3.55"),
        "iso_alpha2": "VN",
        "iso_alpha3": "VNM",
        "region": "Southeast Asia",
    },
    "NG": {
        "country_name": "Nigeria",
        "total_ghg_mt": Decimal("280"),
        "gdp_ppp_b": Decimal("1210"),
        "per_capita_tco2e": Decimal("1.29"),
        "iso_alpha2": "NG",
        "iso_alpha3": "NGA",
        "region": "Africa",
    },
    "EG": {
        "country_name": "Egypt",
        "total_ghg_mt": Decimal("340"),
        "gdp_ppp_b": Decimal("1590"),
        "per_capita_tco2e": Decimal("3.22"),
        "iso_alpha2": "EG",
        "iso_alpha3": "EGY",
        "region": "Africa / Middle East",
    },
    "AE": {
        "country_name": "United Arab Emirates",
        "total_ghg_mt": Decimal("220"),
        "gdp_ppp_b": Decimal("730"),
        "per_capita_tco2e": Decimal("22.44"),
        "iso_alpha2": "AE",
        "iso_alpha3": "ARE",
        "region": "Middle East",
    },
    "IL": {
        "country_name": "Israel",
        "total_ghg_mt": Decimal("72"),
        "gdp_ppp_b": Decimal("500"),
        "per_capita_tco2e": Decimal("7.53"),
        "iso_alpha2": "IL",
        "iso_alpha3": "ISR",
        "region": "Middle East",
    },
    "CL": {
        "country_name": "Chile",
        "total_ghg_mt": Decimal("92"),
        "gdp_ppp_b": Decimal("540"),
        "per_capita_tco2e": Decimal("4.72"),
        "iso_alpha2": "CL",
        "iso_alpha3": "CHL",
        "region": "South America",
    },
    "CO": {
        "country_name": "Colombia",
        "total_ghg_mt": Decimal("178"),
        "gdp_ppp_b": Decimal("880"),
        "per_capita_tco2e": Decimal("3.46"),
        "iso_alpha2": "CO",
        "iso_alpha3": "COL",
        "region": "South America",
    },
    "AR": {
        "country_name": "Argentina",
        "total_ghg_mt": Decimal("345"),
        "gdp_ppp_b": Decimal("1130"),
        "per_capita_tco2e": Decimal("7.53"),
        "iso_alpha2": "AR",
        "iso_alpha3": "ARG",
        "region": "South America",
    },
    "PE": {
        "country_name": "Peru",
        "total_ghg_mt": Decimal("90"),
        "gdp_ppp_b": Decimal("490"),
        "per_capita_tco2e": Decimal("2.68"),
        "iso_alpha2": "PE",
        "iso_alpha3": "PER",
        "region": "South America",
    },
    "NZ": {
        "country_name": "New Zealand",
        "total_ghg_mt": Decimal("78"),
        "gdp_ppp_b": Decimal("260"),
        "per_capita_tco2e": Decimal("15.18"),
        "iso_alpha2": "NZ",
        "iso_alpha3": "NZL",
        "region": "Oceania",
    },
    "PK": {
        "country_name": "Pakistan",
        "total_ghg_mt": Decimal("410"),
        "gdp_ppp_b": Decimal("1510"),
        "per_capita_tco2e": Decimal("1.76"),
        "iso_alpha2": "PK",
        "iso_alpha3": "PAK",
        "region": "South Asia",
    },
    "BD": {
        "country_name": "Bangladesh",
        "total_ghg_mt": Decimal("224"),
        "gdp_ppp_b": Decimal("1230"),
        "per_capita_tco2e": Decimal("1.32"),
        "iso_alpha2": "BD",
        "iso_alpha3": "BGD",
        "region": "South Asia",
    },
}


# =============================================================================
# TABLE 4: GRID EMISSION FACTORS
# =============================================================================

GRID_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Country-level grid EFs (kgCO2e/kWh)
    "US": {"grid_ef_kwh": Decimal("0.417"), "td_loss_pct": Decimal("0.053")},
    "GB": {"grid_ef_kwh": Decimal("0.233"), "td_loss_pct": Decimal("0.077")},
    "DE": {"grid_ef_kwh": Decimal("0.385"), "td_loss_pct": Decimal("0.040")},
    "FR": {"grid_ef_kwh": Decimal("0.052"), "td_loss_pct": Decimal("0.063")},
    "CN": {"grid_ef_kwh": Decimal("0.555"), "td_loss_pct": Decimal("0.055")},
    "IN": {"grid_ef_kwh": Decimal("0.716"), "td_loss_pct": Decimal("0.190")},
    "JP": {"grid_ef_kwh": Decimal("0.457"), "td_loss_pct": Decimal("0.050")},
    "AU": {"grid_ef_kwh": Decimal("0.680"), "td_loss_pct": Decimal("0.050")},
    "CA": {"grid_ef_kwh": Decimal("0.130"), "td_loss_pct": Decimal("0.086")},
    "BR": {"grid_ef_kwh": Decimal("0.074"), "td_loss_pct": Decimal("0.162")},
    "KR": {"grid_ef_kwh": Decimal("0.459"), "td_loss_pct": Decimal("0.036")},
    "ZA": {"grid_ef_kwh": Decimal("0.928"), "td_loss_pct": Decimal("0.096")},
    # eGRID Subregions (US)
    "AKGD": {"grid_ef_kwh": Decimal("0.458"), "td_loss_pct": Decimal("0.053")},
    "AKMS": {"grid_ef_kwh": Decimal("0.238"), "td_loss_pct": Decimal("0.053")},
    "AZNM": {"grid_ef_kwh": Decimal("0.401"), "td_loss_pct": Decimal("0.053")},
    "CAMX": {"grid_ef_kwh": Decimal("0.227"), "td_loss_pct": Decimal("0.053")},
    "ERCT": {"grid_ef_kwh": Decimal("0.396"), "td_loss_pct": Decimal("0.053")},
    "FRCC": {"grid_ef_kwh": Decimal("0.399"), "td_loss_pct": Decimal("0.053")},
    "HIMS": {"grid_ef_kwh": Decimal("0.536"), "td_loss_pct": Decimal("0.053")},
    "HIOA": {"grid_ef_kwh": Decimal("0.649"), "td_loss_pct": Decimal("0.053")},
    "MROE": {"grid_ef_kwh": Decimal("0.584"), "td_loss_pct": Decimal("0.053")},
    "MROW": {"grid_ef_kwh": Decimal("0.458"), "td_loss_pct": Decimal("0.053")},
    "NEWE": {"grid_ef_kwh": Decimal("0.222"), "td_loss_pct": Decimal("0.053")},
    "NWPP": {"grid_ef_kwh": Decimal("0.281"), "td_loss_pct": Decimal("0.053")},
    "NYCW": {"grid_ef_kwh": Decimal("0.254"), "td_loss_pct": Decimal("0.053")},
    "NYLI": {"grid_ef_kwh": Decimal("0.504"), "td_loss_pct": Decimal("0.053")},
    "NYUP": {"grid_ef_kwh": Decimal("0.111"), "td_loss_pct": Decimal("0.053")},
    "PRMS": {"grid_ef_kwh": Decimal("0.659"), "td_loss_pct": Decimal("0.053")},
    "RFCE": {"grid_ef_kwh": Decimal("0.325"), "td_loss_pct": Decimal("0.053")},
    "RFCM": {"grid_ef_kwh": Decimal("0.513"), "td_loss_pct": Decimal("0.053")},
    "RFCW": {"grid_ef_kwh": Decimal("0.496"), "td_loss_pct": Decimal("0.053")},
    "RMPA": {"grid_ef_kwh": Decimal("0.551"), "td_loss_pct": Decimal("0.053")},
    "SPNO": {"grid_ef_kwh": Decimal("0.528"), "td_loss_pct": Decimal("0.053")},
    "SPSO": {"grid_ef_kwh": Decimal("0.449"), "td_loss_pct": Decimal("0.053")},
    "SRMV": {"grid_ef_kwh": Decimal("0.380"), "td_loss_pct": Decimal("0.053")},
    "SRMW": {"grid_ef_kwh": Decimal("0.664"), "td_loss_pct": Decimal("0.053")},
    "SRSO": {"grid_ef_kwh": Decimal("0.432"), "td_loss_pct": Decimal("0.053")},
    "SRTV": {"grid_ef_kwh": Decimal("0.436"), "td_loss_pct": Decimal("0.053")},
}


# =============================================================================
# TABLE 5: BUILDING EUI BENCHMARKS
# =============================================================================

# kgCO2e per sqm per year, by property type and climate zone
BUILDING_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "office": {
        "tropical": Decimal("85.0"),
        "dry": Decimal("72.0"),
        "temperate": Decimal("68.0"),
        "continental": Decimal("92.0"),
        "polar": Decimal("110.0"),
    },
    "retail": {
        "tropical": Decimal("95.0"),
        "dry": Decimal("80.0"),
        "temperate": Decimal("75.0"),
        "continental": Decimal("100.0"),
        "polar": Decimal("120.0"),
    },
    "warehouse": {
        "tropical": Decimal("35.0"),
        "dry": Decimal("30.0"),
        "temperate": Decimal("28.0"),
        "continental": Decimal("40.0"),
        "polar": Decimal("50.0"),
    },
    "hotel": {
        "tropical": Decimal("110.0"),
        "dry": Decimal("95.0"),
        "temperate": Decimal("88.0"),
        "continental": Decimal("115.0"),
        "polar": Decimal("140.0"),
    },
    "hospital": {
        "tropical": Decimal("145.0"),
        "dry": Decimal("125.0"),
        "temperate": Decimal("118.0"),
        "continental": Decimal("155.0"),
        "polar": Decimal("180.0"),
    },
    "residential": {
        "tropical": Decimal("55.0"),
        "dry": Decimal("48.0"),
        "temperate": Decimal("45.0"),
        "continental": Decimal("65.0"),
        "polar": Decimal("85.0"),
    },
}


# =============================================================================
# TABLE 6: VEHICLE EMISSION FACTORS
# =============================================================================

VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "passenger_car": {
        "description": "Average passenger car (ICE)",
        "annual_emissions_kgco2e": Decimal("4600"),
        "annual_distance_km": Decimal("12000"),
        "ef_per_km": Decimal("0.38333"),
        "fuel_type": "mixed",
    },
    "light_commercial": {
        "description": "Light commercial vehicle (van)",
        "annual_emissions_kgco2e": Decimal("6200"),
        "annual_distance_km": Decimal("20000"),
        "ef_per_km": Decimal("0.31000"),
        "fuel_type": "diesel",
    },
    "heavy_commercial": {
        "description": "Heavy commercial vehicle (truck)",
        "annual_emissions_kgco2e": Decimal("18500"),
        "annual_distance_km": Decimal("40000"),
        "ef_per_km": Decimal("0.46250"),
        "fuel_type": "diesel",
    },
    "motorcycle": {
        "description": "Motorcycle / scooter",
        "annual_emissions_kgco2e": Decimal("1800"),
        "annual_distance_km": Decimal("8000"),
        "ef_per_km": Decimal("0.22500"),
        "fuel_type": "petrol",
    },
    "electric_vehicle": {
        "description": "Battery electric vehicle (BEV)",
        "annual_emissions_kgco2e": Decimal("1200"),
        "annual_distance_km": Decimal("12000"),
        "ef_per_km": Decimal("0.10000"),
        "fuel_type": "electric",
    },
}


# =============================================================================
# TABLE 7: EEIO SECTOR FACTORS
# =============================================================================

EEIO_SECTOR_FACTORS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "sector": "Energy",
        "ef_kgco2e_per_usd": Decimal("0.950"),
        "naics_prefix": "211",
        "description": "Oil, gas extraction and support activities",
    },
    "materials": {
        "sector": "Materials",
        "ef_kgco2e_per_usd": Decimal("0.680"),
        "naics_prefix": "31-33",
        "description": "Chemical, metals, paper manufacturing",
    },
    "industrials": {
        "sector": "Industrials",
        "ef_kgco2e_per_usd": Decimal("0.380"),
        "naics_prefix": "332",
        "description": "Fabricated metals, machinery, equipment",
    },
    "consumer_discretionary": {
        "sector": "Consumer Discretionary",
        "ef_kgco2e_per_usd": Decimal("0.180"),
        "naics_prefix": "44-45",
        "description": "Retail trade, consumer durables",
    },
    "consumer_staples": {
        "sector": "Consumer Staples",
        "ef_kgco2e_per_usd": Decimal("0.200"),
        "naics_prefix": "311",
        "description": "Food, beverage, tobacco manufacturing",
    },
    "healthcare": {
        "sector": "Health Care",
        "ef_kgco2e_per_usd": Decimal("0.095"),
        "naics_prefix": "325",
        "description": "Pharmaceutical and medical device manufacturing",
    },
    "financials": {
        "sector": "Financials",
        "ef_kgco2e_per_usd": Decimal("0.025"),
        "naics_prefix": "52",
        "description": "Finance, insurance, real estate services",
    },
    "information_technology": {
        "sector": "Information Technology",
        "ef_kgco2e_per_usd": Decimal("0.068"),
        "naics_prefix": "334",
        "description": "Computer, electronic, semiconductor manufacturing",
    },
    "communication_services": {
        "sector": "Communication Services",
        "ef_kgco2e_per_usd": Decimal("0.060"),
        "naics_prefix": "517",
        "description": "Telecommunications, media, entertainment",
    },
    "utilities": {
        "sector": "Utilities",
        "ef_kgco2e_per_usd": Decimal("1.420"),
        "naics_prefix": "22",
        "description": "Electric power generation, natural gas distribution",
    },
    "real_estate": {
        "sector": "Real Estate",
        "ef_kgco2e_per_usd": Decimal("0.250"),
        "naics_prefix": "531",
        "description": "Real estate leasing, management services",
    },
    "other": {
        "sector": "Other / Unclassified",
        "ef_kgco2e_per_usd": Decimal("0.320"),
        "naics_prefix": "999",
        "description": "Unclassified or diversified sector average",
    },
}


# =============================================================================
# TABLE 8: PCAF DATA QUALITY MATRIX
# =============================================================================

PCAF_DATA_QUALITY_MATRIX: Dict[str, Dict[int, Dict[str, Any]]] = {
    "listed_equity": {
        1: {
            "description": "Verified Scope 1+2 emissions from investee",
            "data_source": "Audited/verified GHG report",
            "uncertainty_pct": Decimal("10"),
            "example": "CDP-verified, third-party audited emissions",
        },
        2: {
            "description": "Unverified reported Scope 1+2 emissions",
            "data_source": "Self-reported GHG data (unaudited)",
            "uncertainty_pct": Decimal("20"),
            "example": "Annual sustainability report, unaudited",
        },
        3: {
            "description": "Estimated using physical activity data",
            "data_source": "Production data + emission factors",
            "uncertainty_pct": Decimal("35"),
            "example": "Energy consumption x grid EF",
        },
        4: {
            "description": "Estimated using revenue-based EEIO",
            "data_source": "Revenue x sector EEIO factor",
            "uncertainty_pct": Decimal("50"),
            "example": "Company revenue x EEIO sector factor",
        },
        5: {
            "description": "Estimated using sector average",
            "data_source": "Sector average emission intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "GICS sector average tCO2e per $M revenue",
        },
    },
    "corporate_bond": {
        1: {
            "description": "Verified Scope 1+2 emissions from issuer",
            "data_source": "Audited/verified GHG report",
            "uncertainty_pct": Decimal("10"),
            "example": "CDP-verified emissions from bond issuer",
        },
        2: {
            "description": "Unverified reported Scope 1+2 emissions",
            "data_source": "Self-reported GHG data",
            "uncertainty_pct": Decimal("20"),
            "example": "Issuer sustainability report, unaudited",
        },
        3: {
            "description": "Estimated using physical activity data",
            "data_source": "Production data + emission factors",
            "uncertainty_pct": Decimal("35"),
            "example": "Energy consumption x grid EF for issuer",
        },
        4: {
            "description": "Estimated using revenue-based EEIO",
            "data_source": "Revenue x sector EEIO factor",
            "uncertainty_pct": Decimal("50"),
            "example": "Issuer revenue x EEIO sector factor",
        },
        5: {
            "description": "Estimated using sector average",
            "data_source": "Sector average emission intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "GICS sector average for issuer industry",
        },
    },
    "private_equity": {
        1: {
            "description": "Verified Scope 1+2 from portfolio company",
            "data_source": "Audited/verified GHG report",
            "uncertainty_pct": Decimal("10"),
            "example": "Third-party audited portfolio company emissions",
        },
        2: {
            "description": "Unverified reported Scope 1+2 from company",
            "data_source": "Self-reported GHG data",
            "uncertainty_pct": Decimal("20"),
            "example": "Portfolio company sustainability report",
        },
        3: {
            "description": "Estimated using physical activity data",
            "data_source": "Production data + emission factors",
            "uncertainty_pct": Decimal("35"),
            "example": "Portfolio company energy use x grid EF",
        },
        4: {
            "description": "Estimated using revenue-based EEIO",
            "data_source": "Revenue x sector EEIO factor",
            "uncertainty_pct": Decimal("50"),
            "example": "Portfolio company revenue x sector EEIO",
        },
        5: {
            "description": "Estimated using sector average",
            "data_source": "Sector average emission intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "Sector avg applied to portfolio company",
        },
    },
    "project_finance": {
        1: {
            "description": "Verified project-level emissions",
            "data_source": "Audited project GHG inventory",
            "uncertainty_pct": Decimal("10"),
            "example": "Project-specific monitored emissions",
        },
        2: {
            "description": "Unverified project emissions data",
            "data_source": "Self-reported project emissions",
            "uncertainty_pct": Decimal("20"),
            "example": "Developer-reported project emissions",
        },
        3: {
            "description": "Estimated using project activity data",
            "data_source": "Project capacity/output x EFs",
            "uncertainty_pct": Decimal("35"),
            "example": "Project MW capacity x grid-specific EF",
        },
        4: {
            "description": "Estimated using project cost and sector EF",
            "data_source": "Project cost x sector average EF",
            "uncertainty_pct": Decimal("50"),
            "example": "Total project cost x energy sector EF",
        },
        5: {
            "description": "Estimated using general sector average",
            "data_source": "Sector average emission intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "Generic project finance sector average",
        },
    },
    "commercial_real_estate": {
        1: {
            "description": "Actual building energy use data",
            "data_source": "Metered energy consumption + local EFs",
            "uncertainty_pct": Decimal("10"),
            "example": "Smart meter data x local grid EF",
        },
        2: {
            "description": "Estimated from EPC or energy audit",
            "data_source": "Energy Performance Certificate data",
            "uncertainty_pct": Decimal("20"),
            "example": "EPC-rated energy use x local grid EF",
        },
        3: {
            "description": "Estimated from floor area and building type",
            "data_source": "Floor area x EUI benchmark",
            "uncertainty_pct": Decimal("35"),
            "example": "Office 10,000 sqm x temperate EUI benchmark",
        },
        4: {
            "description": "Estimated from property value",
            "data_source": "Property value x property type avg",
            "uncertainty_pct": Decimal("50"),
            "example": "Property value proxy for emissions",
        },
        5: {
            "description": "Estimated from portfolio average",
            "data_source": "Average CRE portfolio intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "Generic CRE sector average",
        },
    },
    "mortgage": {
        1: {
            "description": "Actual building energy use data",
            "data_source": "Metered energy consumption + local EFs",
            "uncertainty_pct": Decimal("10"),
            "example": "Smart meter data x local grid EF",
        },
        2: {
            "description": "Estimated from EPC or energy label",
            "data_source": "Energy label/EPC class data",
            "uncertainty_pct": Decimal("20"),
            "example": "EPC A-G label x national EF per label",
        },
        3: {
            "description": "Estimated from floor area and building type",
            "data_source": "Floor area x residential EUI benchmark",
            "uncertainty_pct": Decimal("35"),
            "example": "House 150 sqm x continental EUI benchmark",
        },
        4: {
            "description": "Estimated from property value or avg",
            "data_source": "Property value x residential avg",
            "uncertainty_pct": Decimal("50"),
            "example": "Property value proxy for emissions",
        },
        5: {
            "description": "Estimated from portfolio average",
            "data_source": "Average mortgage portfolio intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "Generic residential sector average",
        },
    },
    "motor_vehicle_loan": {
        1: {
            "description": "Actual vehicle fuel consumption data",
            "data_source": "OBD/telematics fuel data + fuel EFs",
            "uncertainty_pct": Decimal("10"),
            "example": "Telematics-measured fuel use x fuel EF",
        },
        2: {
            "description": "Manufacturer-rated fuel efficiency",
            "data_source": "Vehicle make/model rated kgCO2/km",
            "uncertainty_pct": Decimal("20"),
            "example": "Toyota Camry rated 0.192 kgCO2/km x km",
        },
        3: {
            "description": "Estimated from vehicle type and avg km",
            "data_source": "Vehicle type x average annual emissions",
            "uncertainty_pct": Decimal("35"),
            "example": "Passenger car avg 4600 kgCO2e/yr",
        },
        4: {
            "description": "Estimated from vehicle value proxy",
            "data_source": "Vehicle value x category average EF",
            "uncertainty_pct": Decimal("50"),
            "example": "Vehicle value-based emission estimate",
        },
        5: {
            "description": "Estimated from fleet average",
            "data_source": "National fleet average emissions",
            "uncertainty_pct": Decimal("60"),
            "example": "Generic motor vehicle sector average",
        },
    },
    "sovereign_bond": {
        1: {
            "description": "Country production emissions (official)",
            "data_source": "UNFCCC national inventory + PPP GDP",
            "uncertainty_pct": Decimal("10"),
            "example": "Official GHG inventory / PPP-adjusted GDP",
        },
        2: {
            "description": "Country emissions (international DB)",
            "data_source": "IEA/WRI CAIT database",
            "uncertainty_pct": Decimal("20"),
            "example": "IEA reported national emissions / GDP",
        },
        3: {
            "description": "Estimated using partial country data",
            "data_source": "Partial inventory + extrapolation",
            "uncertainty_pct": Decimal("35"),
            "example": "Energy sector data extrapolated to total",
        },
        4: {
            "description": "Estimated using regional averages",
            "data_source": "Regional GDP-weighted emissions",
            "uncertainty_pct": Decimal("50"),
            "example": "Sub-Saharan Africa regional average",
        },
        5: {
            "description": "Estimated using global average",
            "data_source": "Global average emission intensity",
            "uncertainty_pct": Decimal("60"),
            "example": "Global average tCO2e per $B GDP",
        },
    },
}


# =============================================================================
# TABLE 9: CURRENCY CONVERSION RATES
# =============================================================================

CURRENCY_CONVERSION_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.000"),
    "EUR": Decimal("1.085"),
    "GBP": Decimal("1.265"),
    "JPY": Decimal("0.00667"),
    "CHF": Decimal("1.130"),
    "CAD": Decimal("0.745"),
    "AUD": Decimal("0.660"),
    "CNY": Decimal("0.138"),
    "INR": Decimal("0.0120"),
    "KRW": Decimal("0.000769"),
    "BRL": Decimal("0.200"),
    "MXN": Decimal("0.058"),
    "SGD": Decimal("0.750"),
    "HKD": Decimal("0.128"),
    "SEK": Decimal("0.097"),
}


# =============================================================================
# TABLE 10: SOVEREIGN COUNTRY DATA
# =============================================================================

# Reuses COUNTRY_EMISSION_FACTORS (Table 3) -- accessor method
# provides the same data structured for sovereign bond calculations.


# =============================================================================
# TABLE 11: CARBON INTENSITY BENCHMARKS
# =============================================================================

CARBON_INTENSITY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "sector": "Energy",
        "benchmark_tco2e_per_m_revenue": Decimal("450"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("150"),
        "sda_pathway_2030": Decimal("280"),
        "sda_pathway_2050": Decimal("50"),
    },
    "materials": {
        "sector": "Materials",
        "benchmark_tco2e_per_m_revenue": Decimal("320"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("120"),
        "sda_pathway_2030": Decimal("200"),
        "sda_pathway_2050": Decimal("40"),
    },
    "industrials": {
        "sector": "Industrials",
        "benchmark_tco2e_per_m_revenue": Decimal("180"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("70"),
        "sda_pathway_2030": Decimal("110"),
        "sda_pathway_2050": Decimal("25"),
    },
    "consumer_discretionary": {
        "sector": "Consumer Discretionary",
        "benchmark_tco2e_per_m_revenue": Decimal("85"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("35"),
        "sda_pathway_2030": Decimal("55"),
        "sda_pathway_2050": Decimal("12"),
    },
    "consumer_staples": {
        "sector": "Consumer Staples",
        "benchmark_tco2e_per_m_revenue": Decimal("95"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("40"),
        "sda_pathway_2030": Decimal("60"),
        "sda_pathway_2050": Decimal("15"),
    },
    "healthcare": {
        "sector": "Health Care",
        "benchmark_tco2e_per_m_revenue": Decimal("45"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("18"),
        "sda_pathway_2030": Decimal("28"),
        "sda_pathway_2050": Decimal("8"),
    },
    "financials": {
        "sector": "Financials",
        "benchmark_tco2e_per_m_revenue": Decimal("12"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("5"),
        "sda_pathway_2030": Decimal("8"),
        "sda_pathway_2050": Decimal("2"),
    },
    "information_technology": {
        "sector": "Information Technology",
        "benchmark_tco2e_per_m_revenue": Decimal("32"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("13"),
        "sda_pathway_2030": Decimal("20"),
        "sda_pathway_2050": Decimal("6"),
    },
    "communication_services": {
        "sector": "Communication Services",
        "benchmark_tco2e_per_m_revenue": Decimal("28"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("12"),
        "sda_pathway_2030": Decimal("18"),
        "sda_pathway_2050": Decimal("5"),
    },
    "utilities": {
        "sector": "Utilities",
        "benchmark_tco2e_per_m_revenue": Decimal("680"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("200"),
        "sda_pathway_2030": Decimal("400"),
        "sda_pathway_2050": Decimal("60"),
    },
    "real_estate": {
        "sector": "Real Estate",
        "benchmark_tco2e_per_m_revenue": Decimal("120"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("50"),
        "sda_pathway_2030": Decimal("75"),
        "sda_pathway_2050": Decimal("18"),
    },
    "other": {
        "sector": "Other / Unclassified",
        "benchmark_tco2e_per_m_revenue": Decimal("150"),
        "paris_aligned_tco2e_per_m_revenue": Decimal("60"),
        "sda_pathway_2030": Decimal("95"),
        "sda_pathway_2050": Decimal("20"),
    },
}


# =============================================================================
# TABLE 12: DOUBLE-COUNTING RULES
# =============================================================================

DC_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "DC-INV-001",
        "description": (
            "If the investee is a consolidated subsidiary (equity share >= 50% "
            "or operational control), its Scope 1/2 emissions must NOT be "
            "counted again under Category 15."
        ),
        "check": "consolidation_boundary",
        "threshold_pct": Decimal("50"),
        "action": "exclude",
        "severity": "error",
    },
    {
        "rule_id": "DC-INV-002",
        "description": (
            "Avoid double-counting between Cat 15 (investments) and Cat 1 "
            "(purchased goods/services) when the investee is also a supplier."
        ),
        "check": "supplier_overlap",
        "threshold_pct": Decimal("0"),
        "action": "flag_warning",
        "severity": "warning",
    },
    {
        "rule_id": "DC-INV-003",
        "description": (
            "Avoid double-counting between Cat 15 equity and Cat 15 debt "
            "for the same investee by using consistent EVIC denominators."
        ),
        "check": "equity_debt_overlap",
        "threshold_pct": Decimal("0"),
        "action": "flag_warning",
        "severity": "warning",
    },
    {
        "rule_id": "DC-INV-004",
        "description": (
            "For sovereign bonds, only count production-based emissions "
            "(not consumption-based) to avoid double-counting with "
            "corporate investments in the same country."
        ),
        "check": "sovereign_production_only",
        "threshold_pct": Decimal("0"),
        "action": "enforce",
        "severity": "error",
    },
    {
        "rule_id": "DC-INV-005",
        "description": (
            "Fund-of-funds: apply look-through to underlying holdings; "
            "do not count at both fund level and holding level."
        ),
        "check": "fund_of_funds",
        "threshold_pct": Decimal("0"),
        "action": "flag_warning",
        "severity": "warning",
    },
]


# =============================================================================
# TABLE 13: COMPLIANCE FRAMEWORK RULES
# =============================================================================

COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    "ghg_protocol": {
        "framework_name": "GHG Protocol Scope 3 Standard",
        "category": "Category 15 - Investments",
        "required_fields": [
            "asset_class",
            "outstanding_amount",
            "financed_emissions",
            "attribution_factor",
            "data_quality_score",
        ],
        "scope_coverage": "Scope 1 + Scope 2 of investee (Scope 3 optional)",
        "attribution_method": "PCAF Global GHG Standard",
        "reporting_period": "Annual",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("1"),
    },
    "pcaf": {
        "framework_name": "PCAF Global GHG Accounting Standard",
        "category": "Financed Emissions",
        "required_fields": [
            "asset_class",
            "outstanding_amount",
            "attribution_factor",
            "financed_emissions",
            "data_quality_score",
            "pcaf_score",
        ],
        "scope_coverage": "Scope 1 + Scope 2 mandatory; Scope 3 recommended",
        "attribution_method": "PCAF-specific per asset class",
        "reporting_period": "Annual",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("0"),
    },
    "csrd_esrs": {
        "framework_name": "CSRD ESRS E1 Climate Change",
        "category": "E1-6 Scope 3 Category 15",
        "required_fields": [
            "asset_class",
            "financed_emissions",
            "data_quality_score",
            "methodology_description",
        ],
        "scope_coverage": "Scope 1 + Scope 2 of investee",
        "attribution_method": "GHG Protocol / PCAF",
        "reporting_period": "Annual (financial year)",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("1"),
    },
    "cdp": {
        "framework_name": "CDP Climate Change Questionnaire",
        "category": "C-FS14.1 Financed Emissions",
        "required_fields": [
            "asset_class",
            "outstanding_amount",
            "financed_emissions",
            "methodology",
            "data_quality",
        ],
        "scope_coverage": "Scope 1 + Scope 2 of investee",
        "attribution_method": "PCAF recommended",
        "reporting_period": "Annual",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("0"),
    },
    "sbti": {
        "framework_name": "Science Based Targets initiative",
        "category": "Financial Sector Guidance",
        "required_fields": [
            "asset_class",
            "financed_emissions",
            "portfolio_coverage",
            "target_year",
        ],
        "scope_coverage": "Scope 1 + Scope 2 mandatory; Scope 3 if material",
        "attribution_method": "PCAF or equivalent",
        "reporting_period": "Annual",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("0"),
    },
    "tcfd": {
        "framework_name": "Task Force on Climate-related Financial Disclosures",
        "category": "Metrics and Targets - Financed Emissions",
        "required_fields": [
            "asset_class",
            "financed_emissions",
            "waci",
            "methodology",
        ],
        "scope_coverage": "Scope 1 + Scope 2 at minimum",
        "attribution_method": "PCAF or TCFD recommended methods",
        "reporting_period": "Annual",
        "double_counting_check": False,
        "materiality_threshold_pct": Decimal("5"),
    },
    "iso_14064": {
        "framework_name": "ISO 14064-1:2018",
        "category": "Indirect GHG emissions from investments",
        "required_fields": [
            "asset_class",
            "financed_emissions",
            "uncertainty_estimate",
            "methodology_description",
        ],
        "scope_coverage": "Category 5 indirect emissions",
        "attribution_method": "Equity share or financial control",
        "reporting_period": "Annual or reporting period",
        "double_counting_check": True,
        "materiality_threshold_pct": Decimal("1"),
    },
}


# =============================================================================
# TABLE 14: DQI SCORING CRITERIA
# =============================================================================

DQI_SCORING_CRITERIA: Dict[str, Dict[int, Dict[str, str]]] = {
    "temporal_representativeness": {
        1: {"description": "Data from current reporting year", "score_label": "Excellent"},
        2: {"description": "Data from prior year (1 year lag)", "score_label": "Good"},
        3: {"description": "Data from 2-3 years ago", "score_label": "Fair"},
        4: {"description": "Data from 4-5 years ago", "score_label": "Poor"},
        5: {"description": "Data older than 5 years or unknown age", "score_label": "Very Poor"},
    },
    "geographical_representativeness": {
        1: {"description": "Exact geography (country, region)", "score_label": "Excellent"},
        2: {"description": "Same country, different region", "score_label": "Good"},
        3: {"description": "Same continent or economic bloc", "score_label": "Fair"},
        4: {"description": "Different continent, similar economy", "score_label": "Poor"},
        5: {"description": "Global average or unknown", "score_label": "Very Poor"},
    },
    "technological_representativeness": {
        1: {"description": "Exact technology/process match", "score_label": "Excellent"},
        2: {"description": "Similar technology, same sector", "score_label": "Good"},
        3: {"description": "Related technology, broader sector", "score_label": "Fair"},
        4: {"description": "Generic sector technology proxy", "score_label": "Poor"},
        5: {"description": "Cross-sector average or unknown", "score_label": "Very Poor"},
    },
    "completeness": {
        1: {"description": "All required data available (100%)", "score_label": "Excellent"},
        2: {"description": "Most data available (80-99%)", "score_label": "Good"},
        3: {"description": "Majority available (50-79%)", "score_label": "Fair"},
        4: {"description": "Limited data available (20-49%)", "score_label": "Poor"},
        5: {"description": "Very limited (<20%) or estimated", "score_label": "Very Poor"},
    },
    "reliability": {
        1: {"description": "Third-party verified/audited", "score_label": "Excellent"},
        2: {"description": "Published official statistics", "score_label": "Good"},
        3: {"description": "Self-reported, documented method", "score_label": "Fair"},
        4: {"description": "Self-reported, undocumented", "score_label": "Poor"},
        5: {"description": "Assumed or expert judgement", "score_label": "Very Poor"},
    },
}


# =============================================================================
# TABLE 15: UNCERTAINTY RANGES
# =============================================================================

UNCERTAINTY_RANGES: Dict[int, Dict[str, Any]] = {
    1: {
        "pcaf_score": 1,
        "uncertainty_pct": Decimal("10"),
        "confidence_level": Decimal("0.95"),
        "lower_bound_multiplier": Decimal("0.90"),
        "upper_bound_multiplier": Decimal("1.10"),
        "description": "High confidence: verified emissions data",
    },
    2: {
        "pcaf_score": 2,
        "uncertainty_pct": Decimal("20"),
        "confidence_level": Decimal("0.95"),
        "lower_bound_multiplier": Decimal("0.80"),
        "upper_bound_multiplier": Decimal("1.20"),
        "description": "Good confidence: reported but unverified",
    },
    3: {
        "pcaf_score": 3,
        "uncertainty_pct": Decimal("35"),
        "confidence_level": Decimal("0.95"),
        "lower_bound_multiplier": Decimal("0.65"),
        "upper_bound_multiplier": Decimal("1.35"),
        "description": "Moderate confidence: physical activity estimates",
    },
    4: {
        "pcaf_score": 4,
        "uncertainty_pct": Decimal("50"),
        "confidence_level": Decimal("0.95"),
        "lower_bound_multiplier": Decimal("0.50"),
        "upper_bound_multiplier": Decimal("1.50"),
        "description": "Low confidence: revenue-based EEIO estimates",
    },
    5: {
        "pcaf_score": 5,
        "uncertainty_pct": Decimal("60"),
        "confidence_level": Decimal("0.95"),
        "lower_bound_multiplier": Decimal("0.40"),
        "upper_bound_multiplier": Decimal("1.60"),
        "description": "Very low confidence: sector average estimates",
    },
}


# =============================================================================
# TABLE 16: PORTFOLIO ALIGNMENT THRESHOLDS
# =============================================================================

PORTFOLIO_ALIGNMENT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "well_below_2c": {
        "scenario": "Well Below 2 Degrees C",
        "annual_reduction_rate_pct": Decimal("7.0"),
        "target_year": 2030,
        "interim_reduction_pct": Decimal("50"),
        "net_zero_year": 2050,
        "description": "Paris-aligned 1.5C-2C pathway",
    },
    "below_2c": {
        "scenario": "Below 2 Degrees C",
        "annual_reduction_rate_pct": Decimal("4.2"),
        "target_year": 2030,
        "interim_reduction_pct": Decimal("35"),
        "net_zero_year": 2060,
        "description": "2C-aligned pathway",
    },
    "national_pledges": {
        "scenario": "National Pledges (NDCs)",
        "annual_reduction_rate_pct": Decimal("2.5"),
        "target_year": 2030,
        "interim_reduction_pct": Decimal("20"),
        "net_zero_year": 2070,
        "description": "Current NDC commitments pathway",
    },
    "current_policies": {
        "scenario": "Current Policies",
        "annual_reduction_rate_pct": Decimal("0.5"),
        "target_year": 2030,
        "interim_reduction_pct": Decimal("5"),
        "net_zero_year": None,
        "description": "Business-as-usual scenario (no net zero)",
    },
}


# =============================================================================
# METRICS COLLECTOR HELPER
# =============================================================================

def _get_metrics_collector() -> Any:
    """
    Lazy-import metrics collector to avoid circular imports.

    Returns:
        InvestmentMetrics singleton or None if not available.
    """
    try:
        from greenlang.agents.mrv.investments.metrics import get_metrics_collector
        return get_metrics_collector()
    except (ImportError, Exception) as exc:
        logger.debug("Metrics collector unavailable: %s", exc)
        return None


def _get_provenance_manager() -> Any:
    """
    Lazy-import provenance manager to avoid circular imports.

    Returns:
        ProvenanceManager singleton or None if not available.
    """
    try:
        from greenlang.agents.mrv.investments.provenance import get_provenance_manager
        return get_provenance_manager()
    except (ImportError, Exception) as exc:
        logger.debug("Provenance manager unavailable: %s", exc)
        return None


# =============================================================================
# ENGINE CLASS
# =============================================================================


class InvestmentDatabaseEngine:
    """
    Thread-safe singleton engine for investment emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all
    investment asset classes per the PCAF Global GHG Accounting Standard.
    Every lookup is recorded via Prometheus metrics (gl_inv_factor_selections_total)
    and returns data suitable for provenance hashing.

    This engine does NOT perform any LLM calls. All factors are retrieved
    from validated, frozen constant tables defined in this module.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        _lookup_count: Total number of factor lookups performed
        _lookup_lock: Lock protecting the lookup counter

    Example:
        >>> engine = InvestmentDatabaseEngine()
        >>> rule = engine.get_pcaf_attribution_rule("listed_equity")
        >>> rule["formula"]
        'outstanding / EVIC'
        >>> ef = engine.get_sector_ef("energy")
        >>> ef
        Decimal('450')
    """

    _instance: Optional["InvestmentDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "InvestmentDatabaseEngine":
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

        logger.info(
            "InvestmentDatabaseEngine initialized: "
            "pcaf_asset_classes=%d, sectors=%d, countries=%d, "
            "grid_regions=%d, building_types=%d, vehicle_categories=%d, "
            "eeio_sectors=%d, currencies=%d, dc_rules=%d, frameworks=%d",
            len(PCAF_ATTRIBUTION_RULES),
            len(SECTOR_EMISSION_FACTORS),
            len(COUNTRY_EMISSION_FACTORS),
            len(GRID_EMISSION_FACTORS),
            len(BUILDING_EUI_BENCHMARKS),
            len(VEHICLE_EMISSION_FACTORS),
            len(EEIO_SECTOR_FACTORS),
            len(CURRENCY_CONVERSION_RATES),
            len(DC_RULES),
            len(COMPLIANCE_FRAMEWORK_RULES),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _record_factor_selection(self, source: str, asset_class: str) -> None:
        """
        Record a factor selection in Prometheus metrics.

        Args:
            source: EF source identifier (e.g., "pcaf", "eeio", "egrid")
            asset_class: Asset class (e.g., "listed_equity", "sovereign_bond")
        """
        try:
            metrics = _get_metrics_collector()
            if metrics is not None:
                metrics.record_factor_selection(source=source, asset_class=asset_class)
        except Exception as exc:
            logger.warning("Failed to record factor selection metric: %s", exc)

    def _quantize(self, value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
        """
        Quantize a Decimal value with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision (default 8 decimal places).

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    # =========================================================================
    # TABLE 1: PCAF ATTRIBUTION RULES
    # =========================================================================

    def get_pcaf_attribution_rule(self, asset_class: str) -> Dict[str, Any]:
        """
        Get PCAF attribution rule for a given asset class.

        Returns the complete attribution rule including formula, numerator,
        denominator, PCAF standard reference, scope coverage, and data
        requirements for the specified asset class.

        Args:
            asset_class: One of 8 PCAF asset classes (e.g., "listed_equity",
                "corporate_bond", "private_equity", "project_finance",
                "commercial_real_estate", "mortgage", "motor_vehicle_loan",
                "sovereign_bond").

        Returns:
            Dict with attribution rule details.

        Raises:
            ValueError: If asset_class is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> rule = engine.get_pcaf_attribution_rule("listed_equity")
            >>> rule["formula"]
            'outstanding / EVIC'
        """
        self._increment_lookup()

        key = asset_class.lower().strip()
        rule = PCAF_ATTRIBUTION_RULES.get(key)
        if rule is None:
            raise ValueError(
                f"PCAF attribution rule not found for asset class '{asset_class}'. "
                f"Available: {sorted(PCAF_ATTRIBUTION_RULES.keys())}"
            )

        self._record_factor_selection("pcaf", key)

        logger.debug(
            "PCAF attribution rule lookup: asset_class=%s, formula=%s",
            key,
            rule["formula"],
        )

        return dict(rule)

    # =========================================================================
    # TABLE 2: SECTOR EMISSION FACTORS
    # =========================================================================

    def get_sector_ef(self, sector: str) -> Decimal:
        """
        Get sector emission factor in tCO2e per $M revenue.

        Args:
            sector: GICS sector key (e.g., "energy", "materials", "utilities").

        Returns:
            Emission factor as Decimal (tCO2e/$M revenue).

        Raises:
            ValueError: If sector is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_sector_ef("energy")
            Decimal('450')
        """
        self._increment_lookup()

        key = sector.lower().strip()
        sector_data = SECTOR_EMISSION_FACTORS.get(key)
        if sector_data is None:
            raise ValueError(
                f"Sector emission factor not found for '{sector}'. "
                f"Available: {sorted(SECTOR_EMISSION_FACTORS.keys())}"
            )

        ef = sector_data["ef_tco2e_per_m_revenue"]
        self._record_factor_selection("gics_sector", key)

        logger.debug(
            "Sector EF lookup: sector=%s, ef=%s tCO2e/$M revenue",
            key,
            ef,
        )

        return ef

    def get_sector_ef_detail(self, sector: str) -> Dict[str, Any]:
        """
        Get detailed sector emission factor data.

        Args:
            sector: GICS sector key.

        Returns:
            Dict with full sector data including scope split.

        Raises:
            ValueError: If sector is not found.
        """
        self._increment_lookup()

        key = sector.lower().strip()
        sector_data = SECTOR_EMISSION_FACTORS.get(key)
        if sector_data is None:
            raise ValueError(
                f"Sector not found: '{sector}'. "
                f"Available: {sorted(SECTOR_EMISSION_FACTORS.keys())}"
            )

        self._record_factor_selection("gics_sector", key)
        return dict(sector_data)

    # =========================================================================
    # TABLE 3: COUNTRY EMISSION FACTORS
    # =========================================================================

    def get_country_emissions(self, country_code: str) -> Dict[str, Any]:
        """
        Get country-level emissions data.

        Returns total GHG emissions, GDP PPP, per capita emissions, and
        region for the specified country.

        Args:
            country_code: ISO 3166-1 alpha-2 code (e.g., "US", "CN", "GB").

        Returns:
            Dict with keys: country_name, total_ghg_mt, gdp_ppp_b,
            per_capita_tco2e, region.

        Raises:
            ValueError: If country_code is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> data = engine.get_country_emissions("US")
            >>> data["total_ghg_mt"]
            Decimal('5222')
        """
        self._increment_lookup()

        key = country_code.upper().strip()
        country_data = COUNTRY_EMISSION_FACTORS.get(key)
        if country_data is None:
            raise ValueError(
                f"Country emission data not found for '{country_code}'. "
                f"Available: {sorted(COUNTRY_EMISSION_FACTORS.keys())}"
            )

        self._record_factor_selection("country", key)

        logger.debug(
            "Country emissions lookup: code=%s, name=%s, total_ghg=%s Mt, "
            "gdp_ppp=%s B, per_capita=%s tCO2e",
            key,
            country_data["country_name"],
            country_data["total_ghg_mt"],
            country_data["gdp_ppp_b"],
            country_data["per_capita_tco2e"],
        )

        return dict(country_data)

    # =========================================================================
    # TABLE 4: GRID EMISSION FACTORS
    # =========================================================================

    def get_grid_ef(
        self, country: str, region: Optional[str] = None
    ) -> Decimal:
        """
        Get grid emission factor in kgCO2e/kWh.

        Looks up by eGRID subregion first (if region provided), then
        falls back to country-level grid EF.

        Args:
            country: ISO alpha-2 country code.
            region: Optional eGRID subregion code (e.g., "CAMX", "ERCT").

        Returns:
            Grid emission factor (kgCO2e/kWh).

        Raises:
            ValueError: If neither region nor country is found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_grid_ef("US", "CAMX")
            Decimal('0.22700000')
        """
        self._increment_lookup()

        # Try region first
        if region is not None:
            region_key = region.upper().strip()
            grid_data = GRID_EMISSION_FACTORS.get(region_key)
            if grid_data is not None:
                ef = self._quantize(grid_data["grid_ef_kwh"])
                self._record_factor_selection("egrid", region_key)
                logger.debug(
                    "Grid EF lookup: region=%s, ef=%s kgCO2e/kWh", region_key, ef
                )
                return ef

        # Fall back to country
        country_key = country.upper().strip()
        grid_data = GRID_EMISSION_FACTORS.get(country_key)
        if grid_data is None:
            raise ValueError(
                f"Grid emission factor not found for country='{country}', "
                f"region='{region}'. Available: {sorted(GRID_EMISSION_FACTORS.keys())}"
            )

        ef = self._quantize(grid_data["grid_ef_kwh"])
        self._record_factor_selection("grid", country_key)

        logger.debug(
            "Grid EF lookup: country=%s, ef=%s kgCO2e/kWh", country_key, ef
        )

        return ef

    # =========================================================================
    # TABLE 5: BUILDING EUI BENCHMARKS
    # =========================================================================

    def get_building_benchmark(
        self, property_type: str, climate_zone: str
    ) -> Decimal:
        """
        Get building EUI benchmark in kgCO2e/sqm/year.

        Args:
            property_type: One of "office", "retail", "warehouse", "hotel",
                "hospital", "residential".
            climate_zone: One of "tropical", "dry", "temperate",
                "continental", "polar".

        Returns:
            Building emission benchmark (kgCO2e/sqm/year).

        Raises:
            ValueError: If property_type or climate_zone is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_building_benchmark("office", "temperate")
            Decimal('68.00000000')
        """
        self._increment_lookup()

        ptype = property_type.lower().strip()
        czone = climate_zone.lower().strip()

        type_data = BUILDING_EUI_BENCHMARKS.get(ptype)
        if type_data is None:
            raise ValueError(
                f"Building type not found: '{property_type}'. "
                f"Available: {sorted(BUILDING_EUI_BENCHMARKS.keys())}"
            )

        benchmark = type_data.get(czone)
        if benchmark is None:
            raise ValueError(
                f"Climate zone not found: '{climate_zone}'. "
                f"Available: {sorted(type_data.keys())}"
            )

        result = self._quantize(benchmark)
        self._record_factor_selection("building_eui", ptype)

        logger.debug(
            "Building EUI lookup: type=%s, zone=%s, benchmark=%s kgCO2e/sqm/yr",
            ptype,
            czone,
            result,
        )

        return result

    # =========================================================================
    # TABLE 6: VEHICLE EMISSION FACTORS
    # =========================================================================

    def get_vehicle_ef(self, vehicle_category: str) -> Dict[str, Any]:
        """
        Get vehicle emission factor data.

        Args:
            vehicle_category: One of "passenger_car", "light_commercial",
                "heavy_commercial", "motorcycle", "electric_vehicle".

        Returns:
            Dict with annual_emissions_kgco2e, annual_distance_km,
            ef_per_km, fuel_type.

        Raises:
            ValueError: If vehicle_category is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> data = engine.get_vehicle_ef("passenger_car")
            >>> data["annual_emissions_kgco2e"]
            Decimal('4600')
        """
        self._increment_lookup()

        key = vehicle_category.lower().strip()
        veh_data = VEHICLE_EMISSION_FACTORS.get(key)
        if veh_data is None:
            raise ValueError(
                f"Vehicle category not found: '{vehicle_category}'. "
                f"Available: {sorted(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        self._record_factor_selection("vehicle", key)

        logger.debug(
            "Vehicle EF lookup: category=%s, annual=%s kgCO2e, km=%s",
            key,
            veh_data["annual_emissions_kgco2e"],
            veh_data["annual_distance_km"],
        )

        return dict(veh_data)

    # =========================================================================
    # TABLE 7: EEIO SECTOR FACTORS
    # =========================================================================

    def get_eeio_factor(self, sector: str) -> Decimal:
        """
        Get EEIO emission factor for a given sector.

        Args:
            sector: Sector key (e.g., "energy", "financials", "utilities").

        Returns:
            EEIO factor in kgCO2e per USD.

        Raises:
            ValueError: If sector is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_eeio_factor("energy")
            Decimal('0.95000000')
        """
        self._increment_lookup()

        key = sector.lower().strip()
        eeio_data = EEIO_SECTOR_FACTORS.get(key)
        if eeio_data is None:
            raise ValueError(
                f"EEIO factor not found for sector '{sector}'. "
                f"Available: {sorted(EEIO_SECTOR_FACTORS.keys())}"
            )

        ef = self._quantize(eeio_data["ef_kgco2e_per_usd"])
        self._record_factor_selection("eeio", key)

        logger.debug(
            "EEIO factor lookup: sector=%s, ef=%s kgCO2e/$", key, ef
        )

        return ef

    def get_eeio_factor_detail(self, sector: str) -> Dict[str, Any]:
        """
        Get detailed EEIO factor data including NAICS prefix.

        Args:
            sector: Sector key.

        Returns:
            Dict with full EEIO data.

        Raises:
            ValueError: If sector is not found.
        """
        self._increment_lookup()

        key = sector.lower().strip()
        eeio_data = EEIO_SECTOR_FACTORS.get(key)
        if eeio_data is None:
            raise ValueError(
                f"EEIO factor not found: '{sector}'. "
                f"Available: {sorted(EEIO_SECTOR_FACTORS.keys())}"
            )

        self._record_factor_selection("eeio", key)
        return dict(eeio_data)

    # =========================================================================
    # TABLE 8: PCAF DATA QUALITY MATRIX
    # =========================================================================

    def get_pcaf_quality_criteria(
        self, asset_class: str, score: int
    ) -> Dict[str, Any]:
        """
        Get PCAF data quality criteria for a given asset class and score.

        Args:
            asset_class: PCAF asset class key.
            score: PCAF data quality score (1-5, where 1 is highest quality).

        Returns:
            Dict with description, data_source, uncertainty_pct, example.

        Raises:
            ValueError: If asset_class or score is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> criteria = engine.get_pcaf_quality_criteria("listed_equity", 1)
            >>> criteria["uncertainty_pct"]
            Decimal('10')
        """
        self._increment_lookup()

        key = asset_class.lower().strip()
        quality_data = PCAF_DATA_QUALITY_MATRIX.get(key)
        if quality_data is None:
            raise ValueError(
                f"PCAF quality data not found for asset class '{asset_class}'. "
                f"Available: {sorted(PCAF_DATA_QUALITY_MATRIX.keys())}"
            )

        score_data = quality_data.get(score)
        if score_data is None:
            raise ValueError(
                f"PCAF quality score {score} not found for '{asset_class}'. "
                f"Valid scores: {sorted(quality_data.keys())}"
            )

        self._record_factor_selection("pcaf_quality", key)

        logger.debug(
            "PCAF quality lookup: asset_class=%s, score=%d, uncertainty=%s%%",
            key,
            score,
            score_data["uncertainty_pct"],
        )

        return dict(score_data)

    # =========================================================================
    # TABLE 9: CURRENCY CONVERSION RATES
    # =========================================================================

    def get_currency_rate(self, currency_code: str) -> Decimal:
        """
        Get exchange rate from currency to USD.

        Args:
            currency_code: ISO 4217 currency code (e.g., "EUR", "GBP", "JPY").

        Returns:
            Exchange rate to USD.

        Raises:
            ValueError: If currency_code is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_currency_rate("EUR")
            Decimal('1.08500000')
        """
        self._increment_lookup()

        key = currency_code.upper().strip()
        rate = CURRENCY_CONVERSION_RATES.get(key)
        if rate is None:
            raise ValueError(
                f"Currency rate not found for '{currency_code}'. "
                f"Available: {sorted(CURRENCY_CONVERSION_RATES.keys())}"
            )

        result = self._quantize(rate)

        logger.debug(
            "Currency rate lookup: code=%s, rate=%s to USD", key, result
        )

        return result

    # =========================================================================
    # TABLE 10: SOVEREIGN COUNTRY DATA
    # =========================================================================

    def get_sovereign_data(self, country_code: str) -> Dict[str, Any]:
        """
        Get sovereign country data for sovereign bond calculations.

        This is a convenience accessor that returns the same data as
        get_country_emissions but structured for the sovereign bond
        attribution formula.

        Args:
            country_code: ISO 3166-1 alpha-2 code.

        Returns:
            Dict with country_name, total_ghg_mt, gdp_ppp_b,
            per_capita_tco2e, ghg_per_gdp_ppp.

        Raises:
            ValueError: If country_code is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> data = engine.get_sovereign_data("US")
            >>> data["gdp_ppp_b"]
            Decimal('25460')
        """
        self._increment_lookup()

        key = country_code.upper().strip()
        country_data = COUNTRY_EMISSION_FACTORS.get(key)
        if country_data is None:
            raise ValueError(
                f"Sovereign data not found for '{country_code}'. "
                f"Available: {sorted(COUNTRY_EMISSION_FACTORS.keys())}"
            )

        # Calculate GHG intensity per GDP PPP (tCO2e per $B GDP PPP)
        ghg_per_gdp = Decimal("0")
        if country_data["gdp_ppp_b"] > 0:
            # total_ghg_mt is in megatonnes; gdp_ppp_b is in billions
            # Result: Mt / $B = tCO2e/$  -- needs conversion
            # total_ghg_mt * 1e6 (tonnes) / gdp_ppp_b * 1e9 ($)
            # = total_ghg_mt / gdp_ppp_b * 1e-3 (tCO2e per $)
            # For sovereign bonds: total_ghg (Mt) / GDP_PPP ($B) -> Mt/$B
            ghg_per_gdp = self._quantize(
                country_data["total_ghg_mt"] / country_data["gdp_ppp_b"],
            )

        self._record_factor_selection("sovereign", key)

        result = {
            "country_name": country_data["country_name"],
            "total_ghg_mt": country_data["total_ghg_mt"],
            "gdp_ppp_b": country_data["gdp_ppp_b"],
            "per_capita_tco2e": country_data["per_capita_tco2e"],
            "ghg_per_gdp_ppp": ghg_per_gdp,
            "region": country_data["region"],
        }

        logger.debug(
            "Sovereign data lookup: code=%s, name=%s, ghg/gdp=%s",
            key,
            result["country_name"],
            ghg_per_gdp,
        )

        return result

    # =========================================================================
    # TABLE 11: CARBON INTENSITY BENCHMARKS
    # =========================================================================

    def get_carbon_benchmark(self, sector: str) -> Dict[str, Any]:
        """
        Get carbon intensity benchmark for a sector.

        Returns the current benchmark, Paris-aligned target, and SDA
        pathway values for 2030 and 2050.

        Args:
            sector: Sector key (e.g., "energy", "utilities").

        Returns:
            Dict with benchmark, Paris-aligned, and pathway values.

        Raises:
            ValueError: If sector is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> bm = engine.get_carbon_benchmark("energy")
            >>> bm["paris_aligned_tco2e_per_m_revenue"]
            Decimal('150')
        """
        self._increment_lookup()

        key = sector.lower().strip()
        bm_data = CARBON_INTENSITY_BENCHMARKS.get(key)
        if bm_data is None:
            raise ValueError(
                f"Carbon benchmark not found for sector '{sector}'. "
                f"Available: {sorted(CARBON_INTENSITY_BENCHMARKS.keys())}"
            )

        self._record_factor_selection("benchmark", key)

        logger.debug(
            "Carbon benchmark lookup: sector=%s, benchmark=%s, "
            "paris_aligned=%s tCO2e/$M",
            key,
            bm_data["benchmark_tco2e_per_m_revenue"],
            bm_data["paris_aligned_tco2e_per_m_revenue"],
        )

        return dict(bm_data)

    # =========================================================================
    # TABLE 12: DOUBLE-COUNTING RULES
    # =========================================================================

    def get_dc_rules(self) -> List[Dict[str, Any]]:
        """
        Get all double-counting prevention rules.

        Returns:
            List of double-counting rule dictionaries.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> rules = engine.get_dc_rules()
            >>> len(rules)
            5
        """
        self._increment_lookup()

        logger.debug("DC rules lookup: count=%d", len(DC_RULES))

        return [dict(r) for r in DC_RULES]

    # =========================================================================
    # TABLE 13: COMPLIANCE FRAMEWORK RULES
    # =========================================================================

    def get_framework_rules(self, framework: str) -> Dict[str, Any]:
        """
        Get compliance framework rules.

        Args:
            framework: Framework key (e.g., "ghg_protocol", "pcaf",
                "csrd_esrs", "cdp", "sbti", "tcfd", "iso_14064").

        Returns:
            Dict with framework rules and requirements.

        Raises:
            ValueError: If framework is not found.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> rules = engine.get_framework_rules("pcaf")
            >>> rules["framework_name"]
            'PCAF Global GHG Accounting Standard'
        """
        self._increment_lookup()

        key = framework.lower().strip()
        fw_data = COMPLIANCE_FRAMEWORK_RULES.get(key)
        if fw_data is None:
            raise ValueError(
                f"Framework rules not found for '{framework}'. "
                f"Available: {sorted(COMPLIANCE_FRAMEWORK_RULES.keys())}"
            )

        self._record_factor_selection("compliance", key)

        logger.debug(
            "Framework rules lookup: framework=%s, name=%s",
            key,
            fw_data["framework_name"],
        )

        return dict(fw_data)

    # =========================================================================
    # TABLE 15: UNCERTAINTY RANGES
    # =========================================================================

    def get_uncertainty_pct(self, pcaf_score: int) -> Decimal:
        """
        Get uncertainty percentage for a PCAF data quality score.

        Args:
            pcaf_score: PCAF score (1-5).

        Returns:
            Uncertainty percentage as Decimal.

        Raises:
            ValueError: If pcaf_score is not valid.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> engine.get_uncertainty_pct(1)
            Decimal('10')
        """
        self._increment_lookup()

        unc_data = UNCERTAINTY_RANGES.get(pcaf_score)
        if unc_data is None:
            raise ValueError(
                f"Uncertainty data not found for PCAF score {pcaf_score}. "
                f"Valid scores: {sorted(UNCERTAINTY_RANGES.keys())}"
            )

        logger.debug(
            "Uncertainty lookup: score=%d, pct=%s%%",
            pcaf_score,
            unc_data["uncertainty_pct"],
        )

        return unc_data["uncertainty_pct"]

    def get_uncertainty_range(self, pcaf_score: int) -> Dict[str, Any]:
        """
        Get full uncertainty range data for a PCAF score.

        Args:
            pcaf_score: PCAF score (1-5).

        Returns:
            Dict with uncertainty_pct, confidence_level, bounds, description.

        Raises:
            ValueError: If pcaf_score is not valid.
        """
        self._increment_lookup()

        unc_data = UNCERTAINTY_RANGES.get(pcaf_score)
        if unc_data is None:
            raise ValueError(
                f"Uncertainty range not found for score {pcaf_score}. "
                f"Valid: {sorted(UNCERTAINTY_RANGES.keys())}"
            )

        return dict(unc_data)

    # =========================================================================
    # TABLE 14: DQI SCORING CRITERIA
    # =========================================================================

    def get_dqi_criteria(
        self, dimension: str, score: int
    ) -> Dict[str, str]:
        """
        Get DQI scoring criteria for a given dimension and score.

        Args:
            dimension: DQI dimension (e.g., "temporal_representativeness",
                "geographical_representativeness", "completeness", "reliability",
                "technological_representativeness").
            score: Score level (1-5).

        Returns:
            Dict with description and score_label.

        Raises:
            ValueError: If dimension or score is not found.
        """
        self._increment_lookup()

        dim_key = dimension.lower().strip()
        dim_data = DQI_SCORING_CRITERIA.get(dim_key)
        if dim_data is None:
            raise ValueError(
                f"DQI dimension not found: '{dimension}'. "
                f"Available: {sorted(DQI_SCORING_CRITERIA.keys())}"
            )

        score_data = dim_data.get(score)
        if score_data is None:
            raise ValueError(
                f"DQI score {score} not found for '{dimension}'. "
                f"Valid: {sorted(dim_data.keys())}"
            )

        return dict(score_data)

    # =========================================================================
    # TABLE 16: PORTFOLIO ALIGNMENT THRESHOLDS
    # =========================================================================

    def get_alignment_threshold(self, scenario: str) -> Dict[str, Any]:
        """
        Get portfolio alignment threshold for a scenario.

        Args:
            scenario: Scenario key (e.g., "well_below_2c", "below_2c",
                "national_pledges", "current_policies").

        Returns:
            Dict with alignment threshold data.

        Raises:
            ValueError: If scenario is not found.
        """
        self._increment_lookup()

        key = scenario.lower().strip()
        threshold_data = PORTFOLIO_ALIGNMENT_THRESHOLDS.get(key)
        if threshold_data is None:
            raise ValueError(
                f"Alignment threshold not found for '{scenario}'. "
                f"Available: {sorted(PORTFOLIO_ALIGNMENT_THRESHOLDS.keys())}"
            )

        return dict(threshold_data)

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search_factors(self, query: str) -> List[Dict[str, Any]]:
        """
        Search across all factor tables for matching entries.

        Performs case-insensitive substring search across sector names,
        country names, asset class names, and descriptions.

        Args:
            query: Search query string.

        Returns:
            List of matching factor dicts with source table identified.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> results = engine.search_factors("energy")
            >>> len(results) >= 2
            True
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        matches: List[Dict[str, Any]] = []

        # Search PCAF attribution rules
        for key, rule in PCAF_ATTRIBUTION_RULES.items():
            if query_lower in key or query_lower in rule["description"].lower():
                matches.append({
                    "table": "pcaf_attribution_rules",
                    "key": key,
                    "description": rule["description"],
                    "formula": rule["formula"],
                })

        # Search sector emission factors
        for key, sector in SECTOR_EMISSION_FACTORS.items():
            if query_lower in key or query_lower in sector["description"].lower():
                matches.append({
                    "table": "sector_emission_factors",
                    "key": key,
                    "gics_sector": sector["gics_sector"],
                    "ef_tco2e_per_m_revenue": str(sector["ef_tco2e_per_m_revenue"]),
                })

        # Search country emission factors
        for key, country in COUNTRY_EMISSION_FACTORS.items():
            if (
                query_lower in key.lower()
                or query_lower in country["country_name"].lower()
            ):
                matches.append({
                    "table": "country_emission_factors",
                    "key": key,
                    "country_name": country["country_name"],
                    "total_ghg_mt": str(country["total_ghg_mt"]),
                })

        # Search EEIO sector factors
        for key, eeio in EEIO_SECTOR_FACTORS.items():
            if query_lower in key or query_lower in eeio["description"].lower():
                matches.append({
                    "table": "eeio_sector_factors",
                    "key": key,
                    "sector": eeio["sector"],
                    "ef_kgco2e_per_usd": str(eeio["ef_kgco2e_per_usd"]),
                })

        logger.debug(
            "Factor search: query='%s', matches=%d", query, len(matches)
        )

        return matches

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_lookup_count(self) -> int:
        """
        Get the total number of factor lookups performed.

        Returns:
            Integer count of lookups.
        """
        with self._lookup_lock:
            return self._lookup_count

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database contents.

        Returns:
            Dict with counts of all factor categories.

        Example:
            >>> engine = InvestmentDatabaseEngine()
            >>> summary = engine.get_database_summary()
            >>> summary["pcaf_asset_classes"]
            8
        """
        return {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "pcaf_asset_classes": len(PCAF_ATTRIBUTION_RULES),
            "sectors": len(SECTOR_EMISSION_FACTORS),
            "countries": len(COUNTRY_EMISSION_FACTORS),
            "grid_regions": len(GRID_EMISSION_FACTORS),
            "building_types": len(BUILDING_EUI_BENCHMARKS),
            "climate_zones": 5,
            "vehicle_categories": len(VEHICLE_EMISSION_FACTORS),
            "eeio_sectors": len(EEIO_SECTOR_FACTORS),
            "pcaf_quality_asset_classes": len(PCAF_DATA_QUALITY_MATRIX),
            "pcaf_quality_scores": 5,
            "currencies": len(CURRENCY_CONVERSION_RATES),
            "dc_rules": len(DC_RULES),
            "compliance_frameworks": len(COMPLIANCE_FRAMEWORK_RULES),
            "dqi_dimensions": len(DQI_SCORING_CRITERIA),
            "uncertainty_scores": len(UNCERTAINTY_RANGES),
            "alignment_scenarios": len(PORTFOLIO_ALIGNMENT_THRESHOLDS),
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
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_engine_instance: Optional[InvestmentDatabaseEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_database_engine() -> InvestmentDatabaseEngine:
    """
    Get the singleton InvestmentDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance.

    Returns:
        InvestmentDatabaseEngine singleton instance.

    Example:
        >>> engine = get_database_engine()
        >>> rule = engine.get_pcaf_attribution_rule("listed_equity")
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = InvestmentDatabaseEngine()
        return _engine_instance


def reset_database_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    InvestmentDatabaseEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Reference data tables
    "PCAF_ATTRIBUTION_RULES",
    "SECTOR_EMISSION_FACTORS",
    "COUNTRY_EMISSION_FACTORS",
    "GRID_EMISSION_FACTORS",
    "BUILDING_EUI_BENCHMARKS",
    "VEHICLE_EMISSION_FACTORS",
    "EEIO_SECTOR_FACTORS",
    "PCAF_DATA_QUALITY_MATRIX",
    "CURRENCY_CONVERSION_RATES",
    "CARBON_INTENSITY_BENCHMARKS",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING_CRITERIA",
    "UNCERTAINTY_RANGES",
    "PORTFOLIO_ALIGNMENT_THRESHOLDS",
    # Engine class
    "InvestmentDatabaseEngine",
    # Module-level accessors
    "get_database_engine",
    "reset_database_engine",
]
